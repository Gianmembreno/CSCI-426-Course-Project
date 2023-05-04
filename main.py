import tkinter as tk
from tkinter import ttk
import webbrowser

# Import the necessary modules and functions from the provided code
from calculations import preprocess_text, recommend_talksC, recommend_talksP, recommend_talksB, recommend_talks, df

def on_submit():
    input_text = preprocess_text(user_input.get())
    num_recommendations = int(num_recommendations_spinbox.get())
    recommendation_method = method_combobox.get()

    if recommendation_method == "Cosine Similarity":
        recommendations = recommend_talksC(input_text, df, num_recommendations)
    elif recommendation_method == "Pearson Correlation":
        recommendations = recommend_talksP(input_text, df, num_recommendations)
    elif recommendation_method == "BM25 Score":
        recommendations = recommend_talksB(input_text, df, num_recommendations)
    else:
        recommendations = recommend_talks(input_text, df, num_recommendations)

    for row in recommendations_tree.get_children():
        recommendations_tree.delete(row)

    for _, row in recommendations.iterrows():
        recommendations_tree.insert("", "end", values=(row['speaker_1'], row['title'], row['url'], row.get('cos_sim', '-'), row.get('pea_sim', '-'), row.get('bm25_scores', '-')))

def on_double_click(event):
    item = recommendations_tree.selection()[0]
    url = recommendations_tree.item(item, "values")[2]
    webbrowser.open(url)

app = tk.Tk()
app.title("TED Talk Recommender")

frame = ttk.Frame(app, padding="10")
frame.grid(column=0, row=0)

user_input_label = ttk.Label(frame, text="Enter your text:")
user_input_label.grid(column=0, row=0)

user_input = tk.StringVar()
user_input_entry = ttk.Entry(frame, width=60, textvariable=user_input)
user_input_entry.grid(column=1, row=0)

num_recommendations_label = ttk.Label(frame, text="Number of recommendations:")
num_recommendations_label.grid(column=0, row=1)

num_recommendations_spinbox = ttk.Spinbox(frame, from_=1, to=50, width=5)
num_recommendations_spinbox.grid(column=1, row=1)

method_label = ttk.Label(frame, text="Recommendation method:")
method_label.grid(column=0, row=2)

method_combobox = ttk.Combobox(frame, values=["Cosine Similarity", "Pearson Correlation", "BM25 Score", "All Methods"])
method_combobox.current(0)
method_combobox.grid(column=1, row=2)

submit_button = ttk.Button(frame, text="Get Recommendations", command=on_submit)
submit_button.grid(column=1, row=3)

recommendations_tree = ttk.Treeview(app, columns=("Speaker", "Title", "URL", "Cosine Similarity", "Pearson Correlation", "BM25 Score"), show="headings")
recommendations_tree.heading("Speaker", text="Speaker")
recommendations_tree.heading("Title", text="Title")
recommendations_tree.heading("URL", text="URL")
recommendations_tree.heading("Cosine Similarity", text="Cosine Similarity")
recommendations_tree.heading("Pearson Correlation", text="Pearson Correlation")
recommendations_tree.heading("BM25 Score", text="BM25 Score")
recommendations_tree.column("Speaker", width=150)
recommendations_tree.column("Title", width=250)
recommendations_tree.column("URL", width=300)
recommendations_tree.column("Cosine Similarity", width=150)
recommendations_tree.column("Pearson Correlation", width=150)
recommendations_tree.column("BM25 Score", width=150)

recommendations_tree.grid(column=0, row=1, columnspan=2)

recommendations_tree.bind('<Double-1>', on_double_click)

app.mainloop()

