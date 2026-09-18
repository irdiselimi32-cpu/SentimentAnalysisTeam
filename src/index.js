async function analyzeComment() {
    const comment = document.getElementById("commentInput").value;

    const result = await eel.predict_sentiment(comment)();

    const resultBox = document.getElementById("resultBox");
    const sentimentLabel = document.getElementById("sentimentLabel");
    const confidenceText = document.getElementById("confidenceText");

    resultBox.classList.remove("hidden", "positive", "negative", "neutral");

    sentimentLabel.innerText = result.label;
    confidenceText.innerText = "Confidence: " + result.confidence + "%";

    if (result.label === "Positive") {
        resultBox.classList.add("positive");
    } else if (result.label === "Negative") {
        resultBox.classList.add("negative");
    } else if (result.label === "Neutral") {
        resultBox.classList.add("neutral");
    }
}