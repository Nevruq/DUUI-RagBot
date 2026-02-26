
if __name__ == "__main__":
    import DUUIComponent.DUUIComponent as DD

    """
    Example usage of the DUUI component generator.
    """
    # ============================================
    # KONFIGURATION
    # ============================================
    MODEL_ID = "EvilScript/academic-sentiment-classifier"  # HuggingFace Model ID
    COMPONENT_NAME = "sent"                   # Komponentenname (kurz, ohne Leerzeichen)
    OUTPUT_PATH = "./generated_components"    # Ausgabepfad

    # Generiere Komponente
    result = DD.generate_duui_component(
        model_id=MODEL_ID,
        component_name=COMPONENT_NAME,
        output_path=OUTPUT_PATH,
        verbose=True
    )
