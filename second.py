from transformers import pipeline
import textwrap

# Load the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Ask user to paste the input text
print("Paste the text you'd like to summarize (then press Enter):\n")
text = input()

# Optional: Break text into chunks (if long)
def chunk_text(text, max_chars=1000):
    return textwrap.wrap(text, max_chars)

chunks = chunk_text(text)

# Summarize each chunk
summary_output = []
for chunk in chunks:
    summary = summarizer(
        chunk,
        max_length=80,
        min_length=25,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    summary_output.append(summary[0]['summary_text'])

# Display the final summary
final_summary = " ".join(summary_output)
print("\nSummary:\n", final_summary)