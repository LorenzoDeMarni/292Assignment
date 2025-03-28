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
    features["mean"] = np.mean(segment, axis=0)  # Applies to each axis column in the segment (x,y,z)
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
        features_list=[extract_features(seg) for seg in segment]
        feature_df=features_to_dataframe(features_list)
        predictions=model.predict(feature_df)
        labels = ['walking' if p == 0 else 'jumping' for p in predictions]
        #saves the predictions as a new file
        output_df = pd.DataFrame({'Segment': range(len(labels)), 'Prediction': labels})
        output_file=os.path.splitext(csv_file_path)[0]+"_results.csv"
        output_df.to_csv(output_file,index=False)
        popup(output_df)
    except Exception as e:
        messagebox.showerror("Error", str(e))
#GUI
root=tk.Tk()
root.configure(bg="#ffcce0")
root.geometry("1600x1600")
root.title("Activity Classifier App")
#create main frame
main_frame=Frame(root)
main_frame = Frame(root, bg="#ffcce0")
main_frame.pack(expand=1,fill=BOTH)

#create a canvas
my_canvas=Canvas(main_frame, bg="#ffcce0")
my_canvas.pack(side=LEFT,fill=BOTH,expand=1)

#add a scrollbar to the canvas
my_scrollbar=ttk.Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT,fill=Y,)

#configure the canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>',lambda e:my_canvas.configure(scrollregion=my_canvas.bbox("all")) )

#create another frame for the canvas
second_frame = Frame(root, bg="#ffcce0")
my_canvas.create_window((0,0),window=second_frame,anchor="nw")

content_frame = Frame(second_frame, bg="#ffcce0")
content_frame.pack(expand=1)
content_frame.place(relx=0.5,rely=0.4,anchor="center")

label=tk.Label(second_frame,text="WELCOME TO THE ACTIVITY CLASSIFIER APP",font=('Comic Sans MS',58,'bold'),bg="#ffcce0",fg="#4b0033")
label.pack(padx=20,pady=20)

#button addition
button=tk.Button(second_frame,text="SELECT CSV FILE TO BE CLASSIFIED",font=('Comic Sans MS',40),command=open_csv,fg="#4b0033",activebackground="#8db574",activeforeground="white",highlightthickness=4,
highlightbackground="#4b0033", highlightcolor="#4b0033")
button.pack(padx=30,pady=30)

root.mainloop()
