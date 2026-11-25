const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("result");
const predictButton = form.querySelector("button");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    // Get input values
    const studyHours = Number(
        document.getElementById("studyHours").value
    );

    const attendance = Number(
        document.getElementById("attendance").value
    );

    // Step 1: Validate Study Hours
    if (!Number.isFinite(studyHours) || studyHours < 0 || studyHours > 24) {
        showError("Study hours must be between 0 and 24.");
        return;
    }

    // Step 2: Validate Attendance
    if (!Number.isFinite(attendance) || attendance < 0 || attendance > 100) {
        showError("Attendance must be between 0 and 100.");
        return;
    }

    // Prepare input data
    const studentData = {
        StudyHoursPerDay: studyHours,
        AttendanceRate: attendance
    };

    try {

        // Disable button while processing
        predictButton.disabled = true;
        predictButton.innerText = "Predicting...";

        resultDiv.className = "";
        resultDiv.innerText = "Processing your prediction...";

        // Send request to FastAPI
        const response = await fetch("https://student-prediction-o7yk.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(studentData)
        });

        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }

        const result = await response.json();

        // Display prediction
        const prediction = result.prediction;

        resultDiv.className = prediction.toLowerCase();

        resultDiv.innerHTML = `
            <h2>Prediction: ${prediction}</h2>

            <p>
                Fail Probability:
                ${(result.fail_probability * 100).toFixed(2)}%
            </p>

            <div class="progress-container">
                <div class="progress-bar"
                     style="width: ${result.fail_probability * 100}%">
                </div>
            </div>

            <p>Decision Threshold: ${result.threshold}</p>
        `;

    } catch (error) {

        showError("Unable to connect to backend. Please try again.");

        console.error(error);

    } finally {

        // Enable button again
        predictButton.disabled = false;
        predictButton.innerText = "Predict Result";

    }

});

// Error display function
function showError(message) {

    resultDiv.className = "error";
    resultDiv.innerText = message;

}