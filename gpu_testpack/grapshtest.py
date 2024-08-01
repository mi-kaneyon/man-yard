import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import threading

class LoadTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graphics Benchmark Tester")
        self.root.geometry("1000x900")

        self.info_area = tk.Text(root, height=10, width=80, font=("Helvetica", 14))
        self.info_area.grid(column=0, row=7, columnspan=3, padx=10, pady=10)

        self.benchmark_2d_button = ttk.Button(root, text="2D Draw Benchmark", command=self.run_2d_benchmark)
        self.benchmark_2d_button.grid(column=0, row=12, columnspan=3, pady=20)

        self.benchmark_3d_button = ttk.Button(root, text="3D Draw Benchmark", command=self.run_3d_benchmark)
        self.benchmark_3d_button.grid(column=0, row=13, columnspan=3, pady=20)

        ttk.Label(root, text="Test Duration (minutes)").grid(column=0, row=14, padx=10, pady=10)
        self.duration_slider = tk.Scale(root, from_=1, to=30, orient='horizontal')
        self.duration_slider.grid(column=1, row=14, padx=10, pady=10)

        self.loop_var = tk.BooleanVar()
        self.loop_checkbutton = ttk.Checkbutton(root, text="Unlimited", variable=self.loop_var)
        self.loop_checkbutton.grid(column=2, row=14, padx=10, pady=10)

        self.exit_button = ttk.Button(root, text="Exit", command=self.exit_app)
        self.exit_button.grid(column=0, row=15, columnspan=3, pady=20)

    def run_2d_benchmark(self):
        threading.Thread(target=self._run_2d_benchmark, daemon=True).start()

    def _run_2d_benchmark(self):
        self.info_area.insert(tk.END, "\n2D Draw Benchmark started...\n")
        self.run_2d_draw_benchmark(self.root, self.info_area)

    def run_3d_benchmark(self):
        threading.Thread(target=self._run_3d_benchmark, daemon=True).start()

    def _run_3d_benchmark(self):
        self.info_area.insert(tk.END, "\n3D Draw Benchmark started...\n")
        self.run_3d_draw_benchmark(self.info_area)

    def run_2d_draw_benchmark(self, root, info_area):
        top = tk.Toplevel(root)
        top.geometry("800x600")
        canvas = tk.Canvas(top, width=800, height=600)
        canvas.pack()
        image_path = "texture.jpg"  # Path to the uploaded texture image
        image = Image.open(image_path)
        width, height = image.size

        if self.loop_var.get():
            duration = float('inf')
        else:
            duration = self.duration_slider.get() * 60  # Convert minutes to seconds
        start_time = time.time()
        end_time = start_time + duration

        line_images = []
        for y in range(height):
            line_image = image.crop((0, 0, width, y + 1))
            line_tk_image = ImageTk.PhotoImage(line_image)
            line_images.append(line_tk_image)

        while time.time() < end_time:
            for y, line_tk_image in enumerate(line_images):
                canvas.create_image(0, 0, anchor=tk.NW, image=line_tk_image)
                top.update()
            canvas.delete("all")

        total_time = time.time() - start_time
        info_area.insert(tk.END, f"2D Image Draw Benchmark: {total_time:.2f} seconds\n")
        top.destroy()  # 追加: ウィンドウを閉じる

    def run_3d_draw_benchmark(self, info_area):
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        if self.loop_var.get():
            duration = float('inf')
        else:
            duration = self.duration_slider.get() * 60  # Convert minutes to seconds
        start_time = time.time()
        end_time = start_time + duration

        def create_32_polyhedron():
            vertices = [
                (0.0, 0.0, 1.0), (0.894, 0.0, 0.447), (0.276, 0.851, 0.447),
                (-0.724, 0.526, 0.447), (-0.724, -0.526, 0.447), (0.276, -0.851, 0.447),
                (0.724, 0.526, -0.447), (-0.276, 0.851, -0.447), (-0.894, 0.0, -0.447),
                (-0.276, -0.851, -0.447), (0.724, -0.526, -0.447), (0.0, 0.0, -1.0),
                (1.118, 0.649, 0.0), (-1.118, 0.649, 0.0), (-1.118, -0.649, 0.0),
                (1.118, -0.649, 0.0), (0.447, 1.118, 0.276), (-0.447, 1.118, 0.276),
                (-0.447, -1.118, 0.276), (0.447, -1.118, 0.276), (0.447, 1.118, -0.276),
                (-0.447, 1.118, -0.276), (-0.447, -1.118, -0.276), (0.447, -1.118, -0.276),
                (0.724, 0.526, 0.447), (-0.724, 0.526, 0.447), (-0.724, -0.526, 0.447),
                (0.724, -0.526, 0.447), (0.894, 0.0, 0.447), (-0.894, 0.0, 0.447),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1),
                (1, 24, 25), (1, 25, 26), (1, 26, 27), (1, 27, 24),
                (2, 16, 17), (2, 17, 18), (2, 18, 19), (2, 19, 16),
                (3, 8, 9), (3, 9, 10), (3, 10, 11), (3, 11, 8),
                (4, 12, 13), (4, 13, 14), (4, 14, 15), (4, 15, 12),
                (5, 20, 21), (5, 21, 22), (5, 22, 23), (5, 23, 20),
                (6, 24, 27), (7, 24, 6), (8, 24, 7), (9, 24, 8),
                (10, 24, 9), (11, 24, 10), (12, 24, 11), (13, 24, 12),
                (14, 24, 13), (15, 24, 14), (16, 24, 15), (17, 24, 16),
                (18, 24, 17), (19, 24, 18), (20, 24, 19), (21, 24, 20),
                (22, 24, 21), (23, 24, 22), (25, 1, 0), (26, 2, 1),
                (27, 3, 2), (28, 4, 3), (29, 5, 4), (30, 6, 5),
                (31, 0, 6), (25, 2, 16), (26, 3, 17), (27, 4, 18),
                (28, 5, 19), (29, 0, 16), (30, 1, 17), (31, 2, 18),
                (26, 2, 25), (27, 3, 26), (28, 4, 27), (29, 5, 28),
                (30, 6, 29), (31, 0, 30), (25, 1, 31)
            ]

            colors = [
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1],
                [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
                [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                [0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25],
                [0.25, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0.25],
                [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75]
            ]

            glBegin(GL_TRIANGLES)
            for i, face in enumerate(faces):
                glColor3f(*colors[i % len(colors)])
                for vertex in face:
                    glVertex3f(*vertices[vertex])
            glEnd()

        while time.time() < end_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glRotatef(1, 3, 1, 1)
            create_32_polyhedron()
            pygame.display.flip()
            pygame.time.wait(10)

        total_time = time.time() - start_time
        info_area.insert(tk.END, f"3D Draw Benchmark: {total_time:.2f} seconds\n")
        pygame.quit()

    def exit_app(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LoadTestApp(root)
    root.mainloop()
