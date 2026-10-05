//! Omega's opt-in, versioned conversation wire format.
//!
//! A message is a role ID, literal content IDs, and an end-turn ID. Completed
//! records end with the final assistant end-turn; there is no additional end-of-
//! conversation token. Plain-text pretraining does not insert document endings.
//! Persist the version, resolved IDs and full tokenizer identity with weights.

use crate::Tokens;

pub const CHAT_PROTOCOL_VERSION: &str = "omega-chat-v1";
pub const CHAT_TARGET_OBJECTIVE: &str = "assistant-next-token-omega-chat-v1";
pub const CHAT_CONTROL_TOKENS: [&str; 4] = [
    "<|omega_v1_system|>",
    "<|omega_v1_user|>",
    "<|omega_v1_assistant|>",
    "<|omega_v1_end_turn|>",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatTokenIds {
    pub system: u32,
    pub user: u32,
    pub assistant: u32,
    pub end_turn: u32,
}

impl ChatTokenIds {
    pub fn as_array(self) -> [u32; 4] {
        [self.system, self.user, self.assistant, self.end_turn]
    }

    fn role(self, role: ChatRole) -> u32 {
        match role {
            ChatRole::System => self.system,
            ChatRole::User => self.user,
            ChatRole::Assistant => self.assistant,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedConversation {
    pub ids: Vec<u32>,
    /// Eligibility for next-token targets `ids[1..]`, not input validity.
    /// Assistant content and its end-turn are true; all role IDs are false.
    pub target_mask: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatProtocol {
    ids: ChatTokenIds,
}

impl ChatProtocol {
    /// Whether the saved pipeline explicitly opts into any v1 chat control.
    /// Ordinary vocabulary or non-special added tokens with matching spellings
    /// do not opt in. A true result still requires `from_tokenizer` validation:
    /// partial or malformed registrations must not silently fall back to text.
    pub fn has_registered_controls(tokenizer: &Tokens) -> bool {
        tokenizer
            .t
            .get_added_tokens_decoder()
            .values()
            .any(|token| token.special && CHAT_CONTROL_TOKENS.contains(&token.content.as_str()))
    }

    /// Resolve explicitly registered v1 controls. This does not add tokens to
    /// an existing tokenizer or infer a format from vocabulary size.
    pub fn from_tokenizer(tokenizer: &Tokens) -> Result<Self, String> {
        let pipeline = &tokenizer.t;
        if pipeline.get_padding().is_some() || pipeline.get_truncation().is_some() {
            return Err("Chat protocol requires tokenizer padding and truncation disabled".into());
        }
        if pipeline.get_normalizer().is_some() || pipeline.get_post_processor().is_some() {
            return Err("Chat protocol requires no normalizer or post-processor".into());
        }
        let registered = pipeline.get_added_tokens_decoder();
        let mut ids = [0; 4];
        for (index, spelling) in CHAT_CONTROL_TOKENS.iter().enumerate() {
            let id = tokenizer.token_to_id(spelling).ok_or_else(|| {
                format!("Missing {CHAT_PROTOCOL_VERSION} control token {spelling}; train and freeze an opt-in chat tokenizer before model initialization")
            })?;
            let token = registered.get(&id).ok_or_else(|| {
                format!("Chat control {spelling} must be explicitly registered as special")
            })?;
            if token.content != *spelling
                || !token.special
                || token.normalized
                || token.single_word
                || token.lstrip
                || token.rstrip
            {
                return Err(format!("Invalid special-token registration for {spelling}"));
            }
            if ids[..index].contains(&id) {
                return Err("Chat control IDs must be distinct".into());
            }
            ids[index] = id;
        }
        Ok(Self {
            ids: ChatTokenIds {
                system: ids[0],
                user: ids[1],
                assistant: ids[2],
                end_turn: ids[3],
            },
        })
    }

    pub fn version(&self) -> &'static str {
        CHAT_PROTOCOL_VERSION
    }

    pub fn token_ids(&self) -> ChatTokenIds {
        self.ids
    }

    /// Optional initial system message, then alternating user/assistant turns,
    /// ending in assistant. An empty assistant reply still teaches end-turn.
    pub fn encode_conversation(
        &self,
        tokenizer: &Tokens,
        messages: &[ChatMessage],
    ) -> Result<EncodedConversation, String> {
        self.encode(tokenizer, messages, false)
    }

    /// Encode completed history ending in a user turn, then append exactly one
    /// assistant role ID. No truncation or context-window policy is applied here.
    pub fn encode_prompt(
        &self,
        tokenizer: &Tokens,
        messages: &[ChatMessage],
    ) -> Result<Vec<u32>, String> {
        Ok(self.encode(tokenizer, messages, true)?.ids)
    }

    fn encode(
        &self,
        tokenizer: &Tokens,
        messages: &[ChatMessage],
        prompt: bool,
    ) -> Result<EncodedConversation, String> {
        if Self::from_tokenizer(tokenizer)? != *self {
            return Err("Chat protocol token IDs do not match this tokenizer".into());
        }
        validate_roles(messages, prompt)?;
        // tokenizers calls this flag encode_special_tokens: true treats literal
        // spellings as ordinary model input, instead of matching added specials.
        // Controls are appended as IDs below, never by concatenating text.
        let mut content_pipeline = tokenizer.t.clone();
        content_pipeline.set_encode_special_tokens(true);
        let mut ids = Vec::new();
        let mut eligible = Vec::new();
        for (index, message) in messages.iter().enumerate() {
            ids.push(self.ids.role(message.role));
            eligible.push(false);
            let encoding = content_pipeline
                .encode(message.content.as_str(), false)
                .map_err(|error| format!("Cannot encode chat message {}: {error}", index + 1))?;
            let content = encoding.get_ids();
            if content.iter().any(|id| self.ids.as_array().contains(id)) {
                return Err(format!(
                    "Chat message {} emitted a control ID as literal content",
                    index + 1
                ));
            }
            let decoded = content_pipeline
                .decode(content, false)
                .map_err(|error| format!("Cannot verify literal chat content: {error}"))?;
            if decoded != message.content {
                return Err(format!(
                    "Chat tokenizer does not preserve literal content in message {}",
                    index + 1
                ));
            }
            let supervise = message.role == ChatRole::Assistant;
            ids.extend_from_slice(content);
            eligible.extend(std::iter::repeat_n(supervise, content.len()));
            ids.push(self.ids.end_turn);
            eligible.push(supervise);
        }
        if prompt {
            ids.push(self.ids.assistant);
            eligible.push(false);
        }
        Ok(EncodedConversation {
            ids,
            target_mask: eligible.into_iter().skip(1).collect(),
        })
    }
}

fn validate_roles(messages: &[ChatMessage], prompt: bool) -> Result<(), String> {
    let start = usize::from(messages.first().is_some_and(|m| m.role == ChatRole::System));
    let turns = &messages[start..];
    if turns.is_empty() {
        return Err("Chat requires at least one user turn".into());
    }
    for (index, message) in turns.iter().enumerate() {
        let expected = if index % 2 == 0 {
            ChatRole::User
        } else {
            ChatRole::Assistant
        };
        if message.role != expected {
            return Err(format!(
                "Invalid chat role at message {}: expected {expected:?}; only one optional initial system message is supported",
                start + index + 1
            ));
        }
    }
    let expected_last = if prompt {
        ChatRole::User
    } else {
        ChatRole::Assistant
    };
    if turns.last().expect("nonempty turns").role != expected_last {
        return Err(format!(
            "Chat {} must end with {expected_last:?}",
            if prompt {
                "prompt"
            } else {
                "training conversation"
            }
        ));
    }
    Ok(())
}
