// ==========================================
// WineVision AI
// 3-Input Classification
// ==========================================


// ==========================================
// GET ELEMENTS
// ==========================================

const form =
    document.getElementById("predictionForm");

const predictionText =
    document.getElementById("prediction");

const confidenceText =
    document.getElementById("confidence");

const descriptionText =
    document.getElementById("description");

const modelAccuracy =
    document.getElementById("modelAccuracy");

const predictButton =
    document.querySelector(".predict-btn");


// ==========================================
// LOAD MODEL INFORMATION
// ==========================================

async function loadModelInfo() {

    try {

        const response =
            await fetch("/model-info");


        if (!response.ok) {

            throw new Error(
                "Could not load model information."
            );

        }


        const data =
            await response.json();


        if (modelAccuracy) {

            modelAccuracy.textContent =
                data.accuracy;

        }


    } catch (error) {

        console.error(
            "Model Info Error:",
            error
        );

    }
}


// ==========================================
// GET INPUT DATA
// ==========================================

function getInputData() {

    const alcohol =
        document.getElementById(
            "alcohol"
        ).value;

    const malicAcid =
        document.getElementById(
            "malic_acid"
        ).value;

    const proline =
        document.getElementById(
            "proline"
        ).value;


    return {

        alcohol:
            Number(alcohol),

        malic_acid:
            Number(malicAcid),

        proline:
            Number(proline)

    };
}


// ==========================================
// VALIDATE INPUTS
// ==========================================

function validateInputs(data) {

    if (
        !Number.isFinite(data.alcohol) ||
        data.alcohol <= 0
    ) {

        return false;
    }


    if (
        !Number.isFinite(data.malic_acid) ||
        data.malic_acid <= 0
    ) {

        return false;
    }


    if (
        !Number.isFinite(data.proline) ||
        data.proline <= 0
    ) {

        return false;
    }


    return true;
}


// ==========================================
// DISPLAY RESULT
// ==========================================

function displayResult(data) {

    const confidence =
        Number(data.confidence);


    // Prediction
    predictionText.textContent =
        data.predicted_label;


    // Confidence
    confidenceText.textContent =
        `${confidence.toFixed(2)}%`;


    // Description
    descriptionText.textContent =
        `The Random Forest model classified this wine as ${data.predicted_label} with ${confidence.toFixed(2)}% confidence.`;
}


// ==========================================
// FORM SUBMIT
// ==========================================

form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        // Get input values
        const data =
            getInputData();


        // Validate
        if (!validateInputs(data)) {

            alert(
                "Please enter valid positive values in all fields."
            );

            return;
        }


        // ==================================
        // LOADING STATE
        // ==================================

        predictButton.disabled =
            true;

        predictButton.innerHTML = `
            <span>Analyzing Wine...</span>
            <span class="loading"></span>
        `;


        try {

            // ==================================
            // SEND REQUEST
            // ==================================

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(data)
                    }
                );


            // ==================================
            // HANDLE ERROR
            // ==================================

            if (!response.ok) {

                const errorData =
                    await response.json();

                throw new Error(
                    errorData.detail ||
                    "Prediction failed."
                );
            }


            // ==================================
            // GET RESULT
            // ==================================

            const result =
                await response.json();


            console.log(
                "Prediction:",
                result
            );


            // ==================================
            // DISPLAY RESULT
            // ==================================

            displayResult(
                result
            );


        } catch (error) {

            console.error(
                "Prediction Error:",
                error
            );


            predictionText.textContent =
                "Prediction Failed";


            confidenceText.textContent =
                "--%";


            descriptionText.textContent =
                error.message ||
                "Something went wrong. Please try again.";


        } finally {

            // ==================================
            // RESTORE BUTTON
            // ==================================

            predictButton.disabled =
                false;


            predictButton.innerHTML = `
                <span>Predict Wine Class</span>
                <span class="arrow">→</span>
            `;
        }

    }
);


// ==========================================
// INITIAL LOAD
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadModelInfo();

    }
);