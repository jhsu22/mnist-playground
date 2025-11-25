import customtkinter as ctk


class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.label = ctk.CTkLabel(self, text="Settings Frame")
        self.label.pack(padx=20, pady=20)
