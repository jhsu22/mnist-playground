import customtkinter as ctk

from ui.app import App
from ui.container_frame import BaseFrame


class TrainFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.model_selection_label = ctk.CTkLabel(self, text="Select Model")
        self.model_selection_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.model_selection = ctk.CTkComboBox(self, values=App.MODEL_LIST)
        self.model_selection.grid(row=0, column=1, padx=10, pady=10)

        self.learning_rate_label = ctk.CTkLabel(self, text="Learning Rate")
        self.learning_rate_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.learning_rate_entry = ctk.CTkEntry(self, placeholder_text="0.01")
        self.learning_rate_entry.grid(row=1, column=1, padx=10, pady=10)

        self.epochs_label = ctk.CTkLabel(self, text="Epochs")
        self.epochs_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.epochs_entry = ctk.CTkEntry(self, placeholder_text="10")
        self.epochs_entry.grid(row=2, column=1, padx=10, pady=10)
