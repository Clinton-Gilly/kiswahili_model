import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Expanded dataset
texts = [
    # Respectful/Neutral phrases (labeled as 1)
    "Habari yako?",
    "Asante sana",
    "Karibu",
    "Tafadhali",
    "Pole",
    "Hongera",
    "Ninafurahi kukutana nawe",
    "Nakutakia siku njema",
    "Tunahitaji kuheshimiana",
    "Ninakupenda",
    "Umefanya kazi nzuri",
    "Ninakushukuru kwa msaada wako",
    "Tuko pamoja",
    "Mungu akubariki",
    "Jina lako ni nzuri",
    "Naomba radhi",
    "Ninakutakia mema",
    "Tutaonana tena",
    "Naweza kukusaidia?",
    "Nakupongeza kwa juhudi zako",
    "Leo ni Jumatatu",
    "Jua linawaka leo",
    "Ninaenda sokoni",
    "Chai ina ladha tamu",
    "Kitabu kiko juu ya meza",
    
    # Examples of potentially disrespectful phrases (labeled as 0)
    "Wewe ni mjinga",
    "Sikupendi",
    "Nenda zako",
    "Usinipigie simu tena",
    "Watu wa kabila lako hawana akili, ni wapumbavu tu",
    "Tunapaswa kuondoa hawa wote wasio wa taifa hili",
    "Wewe ni kama punda",
    "Unanuka",
    "Mavi ya kuku",
    "Mjinga wewe",
    "Unanipuuza",
    "Nenda zako hapa",
    "Usiniletee shida",
    "Unanichukiza",
    "Unanifurahisha",
    "Unanisumbua",
    "Usiniletee matatizo",
    "Sitaki kukuona",
    "Wacha kunisumbua",
    "Toka mbele yangu",
    "Usizungumze na mimi tena",
    "Wewe ni mtu mbaya",
    "Sikutaki kamwe",
    "Usinipigie simu tena"
]

# Labels: 1 for respectful/neutral, 0 for potentially disrespectful
labels = [1] * 25 + [0] * 24

# Prepare the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to classify text
def classify_text():
    input_text = text_input.get("1.0", tk.END).strip()
    if input_text:
        vectorized_text = vectorizer.transform([input_text])
        prediction = model.predict(vectorized_text)[0]
        if prediction == 1:
            result = "This text is classified as Respectful/Neutral"
        else:
            result = "This text is classified as Potentially Disrespectful"
        messagebox.showinfo("Classification Result", result)
    else:
        messagebox.showwarning("Warning", "Please enter some text.")

# Function to clear text
def clear_text():
    text_input.delete("1.0", tk.END)

# Create the GUI
root = tk.Tk()
root.title("Kiswahili Text Classifier")
root.configure(bg='lightblue')

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()
frame.configure(bg='lightblue')

label = tk.Label(frame, text="Enter Kiswahili text:")
label.pack()
label.configure(font=("Arial", 12), bg='lightblue')

text_input = tk.Text(frame, height=5, width=50)
text_input.pack()
text_input.configure(font=("Arial", 10))

button_frame = tk.Frame(frame, bg='lightblue')
button_frame.pack(pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_text)
clear_button.pack(side=tk.LEFT, padx=5)
clear_button.configure(font=("Arial", 10), bg='red', fg='white')

submit_button = tk.Button(button_frame, text="Classify", command=classify_text)
submit_button.pack(side=tk.LEFT, padx=5)
submit_button.configure(font=("Arial", 10), bg='green', fg='white')

root.mainloop()