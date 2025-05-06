import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data = {
    "Resume": [
        "Experienced Python developer with Django skills",
        "Data analyst with expertise in SQL and Excel",
        "Cybersecurity expert with knowledge in network security",
        "Machine learning engineer skilled in TensorFlow and NLP",
    ],
    "Category": ["Software Developer", "Data Analyst", "Cybersecurity", "ML Engineer"],
}

df = pd.DataFrame(data)

le = LabelEncoder()
df["Category_Encoded"] = le.fit_transform(df["Category"])

tfidf = TfidfVectorizer(max_features=5000)  
X_tfidf = tfidf.fit_transform(df["Resume"])
y = df["Category_Encoded"]

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svc_model = SVC()
svc_model.fit(X_train, y_train)

pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(svc_model, open("clf.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))  

print("🎉 Model trained and saved successfully!")
