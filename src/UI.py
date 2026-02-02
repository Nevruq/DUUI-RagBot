import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from python_helper_api import python_fix_code


def main() -> None:
    root = tk.Tk()
    root.title("DUUI Python Fix Helper")
    root.geometry("900x600")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)

    header = ttk.Label(root, text="Prompt eingeben:", font=("Segoe UI", 12, "bold"))
    header.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

    input_text = scrolledtext.ScrolledText(root, height=6, wrap=tk.WORD)
    input_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))

    button_frame = ttk.Frame(root)
    button_frame.grid(row=2, column=0, sticky="nw", padx=12, pady=(0, 8))

    output_label = ttk.Label(root, text="Antwort:", font=("Segoe UI", 12, "bold"))
    output_label.grid(row=3, column=0, sticky="w", padx=12, pady=(4, 4))

    output_text = scrolledtext.ScrolledText(root, height=14, wrap=tk.WORD, state=tk.DISABLED)
    output_text.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12))

    root.rowconfigure(4, weight=1)

    def set_output(text: str) -> None:
        output_text.configure(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, text)
        output_text.configure(state=tk.DISABLED)

    def enable_button() -> None:
        submit_button.configure(state=tk.NORMAL)

    def worker(prompt: str) -> None:
        try:
            response = python_fix_code(prompt)
        except Exception as exc:
            response = f"Fehler: {exc}"
        root.after(0, lambda: (set_output(response), enable_button()))

    def on_submit() -> None:
        prompt = input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Hinweis", "Bitte einen Prompt eingeben.")
            return

        submit_button.configure(state=tk.DISABLED)
        set_output("Bitte warten ...")
        threading.Thread(target=worker, args=(prompt,), daemon=True).start()

    submit_button = ttk.Button(button_frame, text="Bestätigen", command=on_submit)
    submit_button.pack(side=tk.LEFT)

    input_text.focus()
    root.mainloop()


if __name__ == "__main__":
    main()
