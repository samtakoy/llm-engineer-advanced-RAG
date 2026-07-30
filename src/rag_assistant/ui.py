"""Gradio-интерфейс поверх RagEngine."""
from typing import List, Tuple

import gradio as gr

from rag_assistant.engine import RagEngine, Source


def format_sources(sources: List[Source]) -> str:
    """Готовит панель найденных фрагментов.

    Аргументы:
        sources: фрагменты, попавшие в контекст ответа.

    Возвращает:
        Текст панели в markdown.
    """
    if not sources:
        return "*Ничего не найдено.*"

    return "\n\n".join(
        f"**{number}. {source.citation}** · score {source.score:.3f}\n\n```\n{source.text}\n```"
        for number, source in enumerate(sources, 1)
    )


def build_app(engine: RagEngine) -> gr.Blocks:
    """Собирает интерфейс чата.

    Аргументы:
        engine: движок вопрос-ответ по документам.

    Возвращает:
        Приложение Gradio, готовое к launch().
    """

    def respond(question: str, history: List[dict]) -> Tuple[str, List[dict], str]:
        """Обрабатывает одно сообщение пользователя."""
        if not question.strip():
            return "", history, ""

        answer = engine.ask(question)
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.text},
        ]
        return "", history, format_sources(answer.sources)

    def reset() -> Tuple[List[dict], str]:
        """Очищает диалог и панель источников."""
        return [], ""

    with gr.Blocks(title = "RAG по годовым отчётам") as app:
        gr.Markdown("## RAG по годовым отчётам\nОтвечает только по документам из папки `documents`.")

        chat_display = gr.Chatbot(label = "Диалог", height = 460)
        question_input = gr.Textbox(
            placeholder = "Задай вопрос и нажми Enter…",
            show_label = False,
        )

        with gr.Row():
            ask_button = gr.Button("Спросить", variant = "primary")
            clear_button = gr.Button("Очистить диалог")

        with gr.Accordion("Найденные фрагменты", open = False):
            sources_panel = gr.Markdown("")

        for trigger in (question_input.submit, ask_button.click):
            trigger(
                respond,
                inputs = [question_input, chat_display],
                outputs = [question_input, chat_display, sources_panel],
            )

        clear_button.click(reset, inputs = [], outputs = [chat_display, sources_panel])

    return app
