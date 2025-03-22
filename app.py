import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import resnet50
import torchvision.models.segmentation as segmentation
import base64

# Charger les variables d'environnement
groq_api_key = "my_api_key" # Utilisez votre clé API Groq ici

# Initialiser le modèle Groq
# Choisir un modèle performant de Groq
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

# Charger un modèle pré-entraîné pour l'analyse d'image (ResNet50)
model = models.resnet50(pretrained=True)
model.eval()

# Fonction de prétraitement de l'image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Définir un modèle de prompt pour LangChain
prompt = PromptTemplate(
    input_variables=["description"],
    template="""Welcome to the AI Medical Image Analysis Assistant! I am an AI developed to assist in the detailed analysis of medical images, helping to provide insights that guide clinical decision-making. Upon uploading an image, I will proceed with a thorough analysis and provide a comprehensive report.

---

### Responsibilities:

1. **Detailed Image Analysis:**
   I will carefully examine the uploaded medical image, looking for abnormalities such as irregularities in shape, texture, and structure, or any signs of disease or health issues. This analysis will include checking for anomalies like masses, fractures, tissue changes, or fluid accumulations.

2. **Findings Report:**
   Once the analysis is complete, I will document all observed abnormalities or concerns. The findings will be presented in a clear, structured format, highlighting the specific areas of concern in the image.

3. **Recommendations and Next Steps:**
   Based on my findings, I will provide specific recommendations for the next steps. This may include suggestions for further diagnostic tests (e.g., CT scans, MRIs), additional imaging, or the need for follow-up appointments with specialists.

4. **Treatment Suggestions (if applicable):**
   If any abnormalities suggest potential health conditions, I will propose possible treatment options, interventions, or management strategies. These suggestions will be based on medical knowledge but should always be verified with a healthcare professional.

---

### Important Notes:

1. **Scope of Response:**  
   I will only analyze and respond to images related to human health. Non-medical images or unrelated content will not be processed.

2. **Clarity of Image:**  
   For accurate analysis, high-quality and clear images are essential. If an image’s quality hinders the analysis, I will inform you and mention which details may not be accurately assessed. It is important to upload clear, well-lit, and focused images for optimal results.

3. **Medical Specificity:**  
   My analysis is based on visual cues within the medical image and should not be considered as a definitive diagnosis. My findings should always be followed up with professional consultation with a healthcare provider, especially for complex or urgent medical conditions.

---

### Disclaimer:
_"Consult with a qualified healthcare provider before making any medical decisions. My analysis aims to guide clinical decisions but is not intended to replace professional medical advice or diagnosis."_

---

### Next Step:
Please upload the medical image you would like to analyze.fournissez des informations : {description}"""
)


# Créer une chaîne LLM
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Configurer la page Streamlit
st.set_page_config(page_title="AI Medical Assistant", page_icon=":robot:")

# Afficher un logo si disponible
logo_path = "imagelogo.jpg"
if os.path.exists(logo_path):
    st.image(logo_path, width=1000)
else:
    st.error("Le fichier 'imagelogo.jpg' n'a pas été trouvé.")

# Ajouter un titre et une description
st.title("X-ray Image Analytics")
st.write("Welcome to the X-ray image AI-powered analysis!")

# Uploader le fichier d'image
uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

# Ajouter un bouton pour lancer l'analyse
submit_button = st.button("Generate analysis")

if submit_button and uploaded_file:
    # Convertir l'image téléchargée en base64
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Prétraiter l'image pour la passer dans le modèle
    st.write("Processing image...")
    input_tensor = preprocess_image(image)
    
    # Effectuer une prédiction avec le modèle pré-entraîné ResNet50
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = f"Predicted class index: {predicted.item()}"  # Vous pouvez mapper cet index à des classes réelles si nécessaire
    
    # Créer une description de l'image
    image_description = f"The uploaded X-ray shows {predicted_label}. Please provide medical insights based on this observation."
    
    # Appel à l'API Groq pour générer la réponse
    try:
        response = llm_chain.run(image_description)
        # Afficher l'analyse générée
        st.write("### X-ray Analysis:")
        st.write(response)
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Groq : {e}")