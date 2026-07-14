// ===============================
// Credit Card Approval Prediction
// script.js
// ===============================

document.addEventListener("DOMContentLoaded", function () {

    // -------------------------------
    // Form Validation
    // -------------------------------

    const form = document.querySelector("form");

    if (form) {

        form.addEventListener("submit", function (e) {

            const inputs = form.querySelectorAll("input[required], select[required]");

            let valid = true;

            inputs.forEach(function (input) {

                if (input.value.trim() === "") {

                    valid = false;

                    input.classList.add("is-invalid");

                } else {

                    input.classList.remove("is-invalid");

                }

            });

            if (!valid) {

                e.preventDefault();

                alert("Please fill all required fields.");

                return;
            }

            // -----------------------
            // Loading Button
            // -----------------------

            const button = form.querySelector("button[type='submit']");

            if (button) {

                button.disabled = true;

                button.innerHTML =
                    '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';

            }

        });

    }

    // -------------------------------
    // Prevent Negative Values
    // -------------------------------

    const numericInputs = document.querySelectorAll("input[type='number']");

    numericInputs.forEach(function (input) {

        input.addEventListener("input", function () {

            if (this.value < 0) {

                this.value = "";

            }

        });

    });

});