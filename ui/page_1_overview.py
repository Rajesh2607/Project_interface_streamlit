import streamlit as st

def render_overview():
    """Page 1: Detailed Project Overview – Problem, Motivation, and System Design"""

    st.title("🧬 DiffusionGenMed")
    st.caption(
        "Generative AI–Driven Synthetic Medical Image Synthesis "
        "for Rare Disease Classification"
    )

    st.markdown("---")

    # =========================================================
    # INTRODUCTION
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>📌 Project Introduction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    DiffusionGenMed is an **end-to-end medical AI system** designed to address one of the
    **most critical challenges in healthcare AI — the lack of sufficient medical imaging data for rare diseases**.

    Unlike common diseases, rare conditions often have **very few labeled CT, MRI, or X-ray scans**.
    This makes it extremely difficult to train reliable AI models, even with modern deep learning techniques.

    This project does **not focus only on classification** but instead focuses on **solving the root cause**:👉 *data scarcity and data imbalance*.
    """)

    # =========================================================
    # PROBLEM STATEMENT (DEEP)
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>🎯 Detailed Problem Statement</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    In real-world medical environments:

    - 🏥 Hospitals may have **only 20–100 images** for a rare disease
    - 📉 Deep learning models trained on such data:
        - Overfit quickly  
        - Fail on unseen cases  
        - Produce misleading confidence scores
    - 🔐 Patient privacy laws (PHI, HIPAA-style constraints) **prevent data sharing**
    - 🧠 Manual diagnosis:
        - Is slow  
        - Depends heavily on expert availability  
        - Is prone to fatigue-related errors

    As a result, **AI systems fail exactly where they are needed the most** — in rare and underrepresented diseases.
    """)

    # =========================================================
    # LIMITATIONS OF EXISTING SOLUTIONS
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>⚠️ Limitations of Existing Approaches</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Several approaches have been attempted to solve this problem, but each has limitations:

    **1. Traditional Machine Learning**
    - Requires large datasets
    - Fails completely for rare diseases

    **2. Data Augmentation (Flip, Rotate, Crop)**
    - Does not create new disease patterns
    - Only reuses existing information

    **3. GAN-Based Image Generation**
    - Suffers from mode collapse
    - Produces visually unrealistic medical images
    - Lacks clinical trust

    **Key Conclusion:**  
    The problem is **not model architecture**, but the **absence of high-quality, diverse training data**.
    """)

    # =========================================================
    # CORE IDEA OF THE PROJECT
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>💡 Core Idea Behind the Project</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Instead of forcing classifiers to learn from insufficient data,
    this project **first learns the distribution of medical images themselves**.

    The central idea is:
    > *If we can generate realistic Syntheticmedical images, we can responsibly increase dataset size and improve diagnosis.*

    To achieve this, we use:
    - **Stable Diffusion** for high-fidelity image synthesis
    - **LoRA (Low-Rank Adaptation)** for efficient medical-domain fine-tuning
    """)

    # =========================================================
    # TWO-MODE SYSTEM DESIGN
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>🔄 Two-Mode System Design (Very Important)</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    This system is intentionally designed with **two clearly separated operational modes**, 
    following real medical AI company standards.

    ### 🔬 Mode 1: Synthetic Data Generation (Research / Training)
    - Used by researchers and AI engineers
    - Inputs:
        - Small set of real medical images
        - Disease-specific text prompts
    - Output:
        - Thousands of **synthetic medical images**
    - ❌ No diagnosis
    - ❌ No patient-level inference

    ### 🩺 Mode 2: Disease Classification (Clinical Decision Support)
    - Used by doctors and radiologists
    - Input:
        - A real medical image (CT / MRI / X-ray)
    - Output:
        - Disease probability scores
        - Confidence level (High / Medium / Low)

    ⚠️ This separation avoids ethical, scientific, and regulatory issues.
    """)

    # =========================================================
    # END-TO-END PIPELINE (DETAILED)
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>🧠 End-to-End System Pipeline</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```
    Step 1: Limited Real Medical Images
            ↓
    Step 2: Data Preprocessing
            - Format standardization
            - Resolution normalization
            - Noise handling
            ↓
    Step 3: Stable Diffusion + LoRA Fine-Tuning
            - Learn disease-specific patterns
            - Preserve base model knowledge
            ↓
    Step 4: Synthetic Medical Image Generation
            - Controlled, prompt-guided synthesis
            - High visual realism
            ↓
    Step 5: Dataset Construction & Balancing
            - Combine real + synthetic images
            - Avoid data leakage
            ↓
    Step 6: Disease Classification Model
            - CNN-based classifier
            ↓
    Step 7: Evaluation & Validation
            - Accuracy, Precision, Recall, F1-score
            - Clinical relevance focus
    ```
    """)

    # =========================================================
    # WHY THIS PROJECT IS IMPORTANT
    # =========================================================
    st.markdown("""
    <div class="card">
    <h2>🌍 Importance and Impact of the Project</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Technical Impact**
    - Solves rare disease data scarcity
    - Demonstrates real generative + discriminative AI integration

    **Clinical Impact**
    - Assists doctors in decision-making
    - Reduces diagnostic delays

    **Academic & Industry Value**
    - Research-grade methodology
    - Company-level system design
    - Strong publication potential

    This project represents a **complete AI lifecycle**,  
    not just a prediction model.
    """)

    st.success("📘 This page fully explains WHAT the project is, WHY it exists, and HOW it works.")
