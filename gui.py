import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, END
import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats
from tkinter import Frame, Canvas, BOTH, LEFT, RIGHT, VERTICAL
from tkinter import ttk

# load the trained model
model = joblib.load("activity_classifier.pkl")

#parameters-window size
window_size=500

#feature extraction
def extract_features(segment):
    features = {}
    # Basic features
    features["mean"] = np.mean(segment, axis=0)
    features["std"] = np.std(segment, axis=0)
    features["min"] = np.min(segment, axis=0)
    features["max"] = np.max(segment, axis=0)
    features["range"] = np.ptp(segment, axis=0)
    features["variance"] = np.var(segment, axis=0)
    features["median"] = np.median(segment, axis=0)
    features["rms"] = np.sqrt(np.mean(np.square(segment), axis=0))
    features["kurtosis"] = stats.kurtosis(segment, axis=0)
    features["skewness"] = stats.skew(segment, axis=0)

    return features

# define feature names
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'variance',
    'median', 'rms', 'kurtosis', 'skewness'
]


def features_to_dataframe(features_list):
    # axes definition
    axes = ['x', 'y', 'z', 'abs']

    # columns
    columns = []
    for feature in feature_names:
        for axis in axes:
            columns.append(f"{feature}_{axis}")

    # create the data rows
    data = []  # list of lists(rows)
    for features in features_list:  
        row = []
        for feature in feature_names:  
            row.extend(features[feature])
        data.append(row)

    #create and return data frame
    df = pd.DataFrame(data, columns=columns)
    return df

#function segmentation
def segment_data_5s(data, window_size):
    segments = []
    for i in range(0, len(data), window_size):
        segment = data.iloc[i:i + window_size, 1:].values
        if len(segment) == window_size:
            segments.append(segment)

    return np.array(segments)

#prediction
def popup(output_df):
    popupwindow=Toplevel(root)
    popupwindow.title("Predictions")
    popupwindow.geometry("400x400")
    popupwindow.configure(bg="#2c3e50")

    text=Text(popupwindow,wrap="word",bg="#2c3e50",fg="#fdf6e3")
    text.pack(side=LEFT,fill=BOTH,expand=1)

    scrollbar = Scrollbar(popupwindow, command=text.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    text.configure(yscrollcommand=scrollbar.set)

    for i, row in output_df.iterrows():
        text.insert(END, f"Segment {row['Segment']}: {row['Prediction']}\n")

def open_csv():
    csv_file_path=filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not csv_file_path:
        return
    try:
        #reads selected csv
        df=pd.read_csv(csv_file_path)
        #passes through the trained model
        segment=segment_data_5s(df,window_size)
        featureslist=[]
        for seg in segment:
            features=extract_features(seg)
            featureslist.append(features)
        feature_df=features_to_dataframe(featureslist)
        abs_cols = [col for col in feature_df.columns if 'abs' in col.lower()]
        feature_df = feature_df[abs_cols]


        predictions=model.predict(feature_df)
        labels=[]
        for p in predictions:
            if p==0:
                labels.append("walking")
            else:
                labels.append("jumping")
        # final classification
        jumping_count = (predictions == 1).sum()
        walking_count = (predictions == 0).sum()
        output_df = pd.DataFrame({'Segment': range(len(labels)), 'Prediction': labels})
        output_df["Segment"] = output_df["Segment"].astype(str)  # Convert entire column to string
        #insert final classification before first row
        output_df.loc[len(output_df)] = ['Final Classification', 'Jumping' if jumping_count > walking_count else 'Walking']
        
        # Clear previous predictions and display new ones
        predictions_text.delete(1.0, END)
        for i, row in output_df.iterrows():
            predictions_text.insert(END, f"Segment {row['Segment']}: {row['Prediction']}\n")
            
    except Exception as e:
        messagebox.showerror("Error", str(e))

#GUI
root=tk.Tk()
root.configure(bg="#1a1f2c")  # Dark blue background
root.geometry("1100x600")  # Smaller window size
root.resizable(False, False)  # Make window non-resizable
root.title("Activity Classifier App")

#create main frame
main_frame=Frame(root)
main_frame = Frame(root, bg="#1a1f2c")
main_frame.pack(expand=1,fill=BOTH)

#create a canvas
my_canvas=Canvas(main_frame, bg="#1a1f2c")
my_canvas.pack(side=LEFT,fill=BOTH,expand=1)

#add a scrollbar to the canvas
my_scrollbar=ttk.Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT,fill=Y,)

#configure the canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>',lambda e:my_canvas.configure(scrollregion=my_canvas.bbox("all")) )

#create another frame for the canvas
second_frame = Frame(root, bg="#1a1f2c")
my_canvas.create_window((0,0),window=second_frame,anchor="nw")

content_frame = Frame(second_frame, bg="#1a1f2c")
content_frame.pack(expand=1)
content_frame.place(relx=0.5,rely=0.4,anchor="center")

label=tk.Label(second_frame,text="WELCOME TO THE ACTIVITY CLASSIFIER APP",font=('Comic Sans MS',33,'bold'),bg="#1a1f2c",fg="#ffffff")
label.pack(padx=20,pady=20)

#button addition
button=tk.Button(second_frame,text="SELECT CSV FILE TO BE CLASSIFIED",font=('Comic Sans MS',20),command=open_csv,fg="#ffffff",bg="#2c3e50",activebackground="#34495e",activeforeground="white",highlightthickness=4,
highlightbackground="#3498db", highlightcolor="#3498db")
button.pack(padx=30,pady=30)

# Add text area for predictions
predictions_frame = Frame(second_frame, bg="#1a1f2c")
predictions_frame.pack(padx=20, pady=20, fill=BOTH, expand=True)

predictions_label = tk.Label(predictions_frame, text="Classification Results:", font=('Comic Sans MS', 20, 'bold'), bg="#1a1f2c", fg="#ffffff")
predictions_label.pack(pady=10)

predictions_text = Text(predictions_frame, wrap="word", font=('Comic Sans MS', 12), bg="#2c3e50", fg="#ffffff", height=8)
predictions_text.pack(side=LEFT, fill=BOTH, expand=True)

predictions_scrollbar = Scrollbar(predictions_frame, command=predictions_text.yview)
predictions_scrollbar.pack(side=RIGHT, fill=Y)
predictions_text.configure(yscrollcommand=predictions_scrollbar.set)

root.mainloop()