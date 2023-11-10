import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Ayarları
st.set_page_config(
    page_title="Heart Disease",
    page_icon="https://icon-icons.com/icon/among-us-player-red/156942",
    layout="centered",
    menu_items={
        "Get help":"mailto:merve.kandil@turktelekom.com.tr  ",
        "About": "# This is a header. This is an *extremely* cool app!"
        
    }
)

# Başlık Ekleme
st.title("Heart Disease Project")

# Markdown Oluşturma
st.markdown("In this project, we investigated the causes of heart diseases.Based on the reasons we found, we made predictions such as **:red[the patient has heart disease ]** or **:green[the patient does not have heart disease]**.")


# Resim Ekleme
st.image("https://www.allinahealth.org/-/media/hsg-images/chest-pain-causes-and-symptoms-for-a-heart-attack.jpg")


# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **age**")
st.markdown("- **sex**")
st.markdown("- **cp**: chest pain type (4 values)")
st.markdown("- **trestbps**: resting blood pressure")
st.markdown("- **chol**: serum cholestoral in mg/dl")
st.markdown("- **fbs**: fasting blood sugar > 120 mg/dl")
st.markdown("- **restecg**: resting electrocardiographic results (values 0,1,2)")
st.markdown("- **thalach**: maximum heart rate achieved")
st.markdown("- **exang**: exercise induced angina")
st.markdown("- **oldpeak**: ST depression induced by exercise relative to rest")
st.markdown("- **slope**: the slope of the peak exercise ST segment")
st.markdown("- **ca**: number of major vessels (0-3) colored by flourosopy")
st.markdown("- **thal**: 0 = normal; 1 = fixed defect; 2 = reversable defect.The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.")
st.markdown("- **target**: **:red[the patient has heart disease ]** or **:green[the patient does not have heart disease]**.")

# Pandasla veri setini okuyalım
df = pd.read_csv("datasetheart2.csv")

# Küçük bir düzenleme :)
df.oldpeak = df.oldpeak.astype(float)



# Tablo Ekleme
st.table(df.sample(5, random_state=42))

#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Lets!** Find out if you may have heart disease.")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
ca1=st.sidebar.checkbox("Are there any problems with only one vein?")
ca2=st.sidebar.checkbox("Are there any problems with more than one veins")
cp2=st.sidebar.checkbox("Do you have atypical angina chest pain?")
cp3=st.sidebar.checkbox("Do you have non-anginal chest pain?")
sex=st.sidebar.checkbox("Are you man?")

#---------------------------------------------------------------------------------------------------------------------

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

logreg_model = load('logreg_model.pkl')

input_df = pd.DataFrame({
    
    'Ca1': [ca1],
    'Ca2': [ca2],
    'Cp2': [cp2],
    'Cp3': [cp3],
    'Sex': [sex]
})

pred = logreg_model.predict(input_df.values)
pred_probability = np.round(logreg_model.predict_proba(input_df.values), 2)

#print(pred),
#print(pred_probability),
#print( pred_probability[:,:1])
#print(pred_probability[:,1:])

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'Ca1': [ca1],
    'Ca2': [ca2],
    'Cp2': [cp2],
    'Cp3': [cp3],
    'Sex': [sex],
    'Prediction': [pred],
    'You do not have heart disease!Probability': str(pred_probability[:,:1][0][0]*100)+'%',
    'You may have heart disease!Probability': str(pred_probability[:,1:][0][0]*100)+'%'
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","You don't have heart disease! "))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","You may have heart disease!  "))
     
    st.table(results_df)

    if pred == 0:
        
        st.image("https://qph.cf2.quoracdn.net/main-qimg-d309a621b1225e33605772944a491989-lq")
           
    else:
        st.image("https://www.metropolisindia.com/upgrade/blog/upload/2021/03/Blog-10-Banner-1.png")
else:
    st.markdown("Please click the *Submit Button*!")