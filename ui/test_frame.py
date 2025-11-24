import customtkinter as ctk


class TestFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.label = ctk.CTkLabel(self, text="Test Frame")
        self.label.pack(padx=20, pady=20)
