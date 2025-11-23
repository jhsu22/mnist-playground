"""
Main UI application
"""

import customtkinter as ctk


class MnistPlayground(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("MNIST Playground")
        self.geometry("1200x800")

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.title = ctk.CTkLabel(self, text="MNIST Playground", padx=20, pady=10)
        self.title.grid(row=0, column=0, sticky="nsew")


if __name__ == "__main__":
    app = MnistPlayground()
    app.mainloop()
