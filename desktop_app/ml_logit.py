import tkinter.messagebox
import customtkinter
import pandas as pd
from tkinter import ttk
from libs.model import predict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
from libs.model import map_labels_to_numbers, predict

customtkinter.set_appearance_mode('System')
customtkinter.set_default_color_theme('blue')

class TitanicApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title('Titanic ML App')
        self.geometry(f"{1200}x{700}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2,3), weight=0)
        self.grid_rowconfigure((0,1,2), weight=0)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky='nsew')
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text='Titanic ML App 1.0', font=customtkinter.CTkFont(size=20, weight='bold'))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button1 = customtkinter.CTkButton(self.sidebar_frame, text='Predict', command=self.get_prediction)
        self.sidebar_button1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button2 = customtkinter.CTkButton(self.sidebar_frame, text='Load CSV', command=self.load_csv)
        self.sidebar_button2.grid(row=2, column=0, padx=20, pady=10)

        self.apperance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text='Apperance mode:', anchor='w')
        self.apperance_mode_label.grid(row=3, column=0, padx=20, pady=(10,0))
        self.apperance_mode_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=['Dark', 'Light', 'System'], command=self.change_appearance_mode_event)
        self.apperance_mode_optionmenu.grid(row=4, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text='UI scaling:', anchor='w')
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10,0))
        self.scaling_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=['80%', '90%', '100%', '110%', '120%'], command=self.change_scaling_event)
        self.scaling_optionmenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        self.pclass = customtkinter.CTkEntry(self, placeholder_text='Pclass')
        self.pclass.grid(row=3, column=1, columnspan=1, padx=(20,0), pady=(20,20), sticky='nsew')
        self.age = customtkinter.CTkEntry(self, placeholder_text='Age')
        self.age.grid(row=4, column=1, columnspan=1, padx=(20,0), pady=(0,20), sticky='nsew')
        self.sibsp = customtkinter.CTkEntry(self, placeholder_text='SibSp')
        self.sibsp.grid(row=5, column=1, columnspan=1, padx=(20,0), pady=(20,20), sticky='nsew')
        self.parch = customtkinter.CTkEntry(self, placeholder_text='Parch')
        self.parch.grid(row=6, column=1, columnspan=1, padx=(20,0), pady=(0,20), sticky='nsew')
        self.fare = customtkinter.CTkEntry(self, placeholder_text='Fare')
        self.fare.grid(row=7, column=1, columnspan=1, padx=(20,0), pady=(20,20), sticky='nsew')
        self.embarked = customtkinter.CTkEntry(self, placeholder_text='Embarked')
        self.embarked.grid(row=8, column=1, columnspan=1, padx=(20,0), pady=(0,20), sticky='nsew')
        self.gender = customtkinter.CTkEntry(self, placeholder_text='Gender')
        self.gender.grid(row=9, column=1, columnspan=1, padx=(20,0), pady=(20,20), sticky='nsew')

        self.csvtable = ttk.Treeview(master=self, show='headings')
        self.csvtable.grid(row=0, column=1, padx=(20,0), pady=(20,0), sticky='nsew')

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace('%', '')) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def load_csv(self):
            file_path = 'data/DSP_6.csv'
            try:
                df = pd.read_csv(file_path)
                self.display_csv_table(df)
            except Exception as e:
                tkinter.messagebox.showerror('Error', f'Error loading CSV file: {str(e)}')


    def display_csv_table(self, df):
        for item in self.csvtable.get_children():
            self.csvtable.delete(item)

        columns = list(df.columns)
        self.csvtable["columns"] = columns
        for col in columns:
            self.csvtable.heading(col, text=col)
            self.csvtable.column(col, anchor="center", width=70)

        for index, row in df.iterrows():
            self.csvtable.insert("", 'end', values=list(row))

    def get_prediction(self):
        try:
            pclass = map_labels_to_numbers(self.pclass.get(), {'First': 0, 'Second': 1, 'Third': 2})
            embarked = map_labels_to_numbers(self.embarked.get(), {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2})
            sex = map_labels_to_numbers(self.gender.get(), {'Female': 0, 'Male': 1})
            
            age = float(self.age.get())
            sibsp = float(self.sibsp.get())
            parch = float(self.parch.get())
            fare = float(self.fare.get())

        except ValueError:
            tkinter.messagebox.showerror('Error', 'Please enter valid values for all input fields.')
            return

        data_for_prediction = [[pclass, age, sibsp, parch, fare, embarked, sex]]

        try:
            result = predict(data_for_prediction, 'ml_models/model.pkl')
            prediction = "Survived" if result == 1 else "Not Survived"
        except Exception as e:
            tkinter.messagebox.showerror('Error', f'Error during prediction: {str(e)}')
            return

        tkinter.messagebox.showinfo('Prediction Result', f'The predicted outcome is: {prediction}')



if __name__ == '__main__':
    app = TitanicApp()
    app.mainloop()
