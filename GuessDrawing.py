import tkinter as tk
import numpy as np

class DigitDrawer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw Digit 28x28")

        self.pixel_size = 10
        self.grid_size = 28
        self.canvas_size = self.pixel_size * self.grid_size

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.pixels = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.draw_grid_lines()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=0, column=0, padx=5)

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=0, column=1, padx=5)

        self.prediction_label = tk.Label(self.root, text="Draw a digit and click Predict")
        self.prediction_label.pack()

        # Load your model weights here
        data = np.load("weights.npz")
        self.Wx = data["Wx"]
        self.Wa = data["Wa"]
        self.Wb = data["Wb"]
        self.Bx = data["Bx"]
        self.Ba = data["Ba"]
        self.Bb = data["Bb"]

    def sigmoid(self, x):
        return np.maximum(0, x)  # ReLU

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def draw_grid_lines(self):
        for i in range(self.grid_size + 1):
            pos = i * self.pixel_size
            self.canvas.create_line(pos, 0, pos, self.canvas_size, fill="#ddd")
            self.canvas.create_line(0, pos, self.canvas_size, pos, fill="#ddd")

    def draw(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.pixels[y, x] = 1
            x1 = x * self.pixel_size
            y1 = y * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")

    def clear(self):
        self.pixels.fill(0)
        self.canvas.delete("all")
        self.draw_grid_lines()
        self.prediction_label.config(text="Draw a digit and click Predict")

    def predict(self):
        # Convert to float and normalize like training data
        input_data = self.pixels.astype(np.float32).flatten()  # shape (784,)
        # Your training used values scaled between 0-1 by dividing by 255, so we match that
        # Here pixels are 0 or 1, so already 0 or 1. You can multiply by 1.0 to get float
        input_data = input_data.reshape(1, -1)

        # Forward pass through your network
        Z1 = np.dot(input_data, self.Wx) + self.Bx
        A = self.sigmoid(Z1)
        Z2 = np.dot(A, self.Wa) + self.Ba
        B = self.sigmoid(Z2)
        Z3 = np.dot(B, self.Wb) + self.Bb
        output = self.softmax(Z3.flatten())

        predicted_digit = np.argmax(output)
        confidence = output[predicted_digit]

        self.prediction_label.config(text=f"Prediction: {predicted_digit} (Confidence: {confidence:.2f})")
        print(f"Prediction: {predicted_digit} (Confidence: {confidence:.2f})")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitDrawer()
    app.run()