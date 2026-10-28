import os
import gradio as gr
from interactive_sampler import create_ui, CUSTOM_CSS

demo = create_ui()
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=CUSTOM_CSS,
        share=share,
    )
