
def truncate_text_based_on_stop_sequence(text: str, stop_sequences: list[str]):
    for stop_sequence in stop_sequences:
        stop_index = text.find(stop_sequence)
        if stop_index != -1:
            text = text[:stop_index + len(stop_sequence)]
    return text