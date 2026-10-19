"""Typed Ollama exceptions with actionable messages (Turkish UI strings)."""

from __future__ import annotations


class OllamaError(Exception):
    """Base class; carries a user-facing message and a technical detail."""

    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class OllamaConnectionError(OllamaError):
    def __init__(self, base_url: str, detail: str | None = None) -> None:
        super().__init__(
            "Ollama sunucusuna bağlanılamadı. Ollama'nın çalıştığından emin olun "
            "(örn. Terminalde `ollama serve`) ve adresi kontrol edin.",
            detail or f"base_url={base_url}",
        )
        self.base_url = base_url


class OllamaTimeoutError(OllamaError):
    def __init__(self, timeout_s: float, detail: str | None = None) -> None:
        super().__init__(
            f"Ollama isteği zaman aşımına uğradı ({timeout_s:.0f} sn). "
            "Model büyükse veya ilk yüklenme uzun sürüyorsa zaman aşımını artırın.",
            detail,
        )


class OllamaModelNotFoundError(OllamaError):
    def __init__(self, model: str, detail: str | None = None) -> None:
        super().__init__(
            f"Seçilen model Ollama'da kurulu değil: {model}. "
            "Modeli `ollama pull` ile indirin veya kurulu bir model seçin.",
            detail,
        )


class OllamaResponseError(OllamaError):
    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message, detail)


class OllamaInvalidOutputError(OllamaError):
    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message, detail)


class OllamaContextTooLongError(OllamaError):
    def __init__(self, original: str, detail: str | None = None) -> None:
        super().__init__(
            "İstek modelin bağlam penceresini aştı. Girdiyi kısaltın ya da "
            "daha yüksek num_ctx değerine sahip bir model seçin.",
            detail or original,
        )


class OllamaRequestCancelled(OllamaError):
    def __init__(self) -> None:
        super().__init__("İstek kullanıcı tarafından iptal edildi.")
