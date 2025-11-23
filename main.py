import customtkinter as ctk

from ui.app import MnistPlayground


def main():
    # Global application configuration
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("assets/themes/dark.json")

    # Create the application window
    app = MnistPlayground()

    # Run the application
    app.mainloop()


if __name__ == "__main__":
    main()
