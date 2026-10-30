import streamlit as st
from detection.had_detector import HADDetector

detector = HADDetector()

st.title("HAD Regex Vulnerability Detector")

pattern = st.text_input("Enter a regex:")

if st.button("Analyze"):
    if pattern:
        report = detector.analyze(pattern)

        st.subheader("Result")
        st.write(f"Vulnerable: {report.is_vulnerable}")
        st.write(f"Complexity: {report.worst_complexity}")

        if report.layer0:
            st.write("Layer 0 detected")
        if report.layer1:
            st.write("Layer 1 detected")
        if report.layer2:
            st.write("Layer 2 detected")

    else:
        st.warning("Please enter a regex")