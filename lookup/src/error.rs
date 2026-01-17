use alloc::string::String;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LookupError {
    #[error("Global cumulative mismatch: {0:?}")]
    GlobalCumulativeMismatch(Option<String>),

    #[error("lookup error: {0}")]
    Msg(String),
}

impl LookupError {
    pub fn msg<M: Into<String>>(m: M) -> Self {
        Self::Msg(m.into())
    }
}
