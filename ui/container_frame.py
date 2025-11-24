import customtkinter as ctk

from config import App


class BaseFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        self.title_frame = ctk.CTkFrame(self, fg_color="#101010")
        self.title_frame.grid(row=0, column=0, sticky="nsew")

        self.title = ctk.CTkLabel(self.title_frame, text=App.TITLE, font=App.FONT_TITLE)
        self.title.grid(row=0, column=0, padx=10, ipady=5, pady=5)

        self.header_frame = ctk.CTkFrame(self, fg_color="#101010")
        self.header_frame.grid(row=1, column=0, sticky="nsew")

        self.train_button = ctk.CTkButton(
            self.header_frame,
            text="Edit Model",
            command=lambda: parent.raise_frame(parent.train_ui),
        )
        self.train_button.grid(row=0, column=0, padx=10, pady=10)

        self.test_button = ctk.CTkButton(
            self.header_frame,
            text="Test Model",
            command=lambda: parent.raise_frame(parent.test_ui),
        )
        self.test_button.grid(row=0, column=1, padx=10, pady=10)

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=2, column=0, sticky="nsew")
