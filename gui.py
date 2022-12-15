# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tkinter as tk
from tkinter import *
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class DemoGUI:

    def update_canvas(self):
        pil_im = Image.fromarray(self.frame_rgb_canvas).resize((480, 480))
        self.photo = ImageTk.PhotoImage(image=pil_im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def __init__(self):
        super().__init__()

        self.root = tk.Tk()
        self.root.title("GUI")

        self.root.geometry("500x630")

        self.notebook = ttk.Notebook(self.root)

        self.canvas = Canvas(self.root, width=480, height=480)
        self.canvas.pack(expand=True)
        self.frame_rgb_canvas = np.zeros([480, 640, 3]).astype(np.uint8)
        self.update_canvas()

        self.TAB1 = Frame(self.notebook, bg="#00766c")
        self.notebook.add(self.TAB1, text='Record')
        self.notebook.bind("<<NotebookTabChanged>>", self.tab_btn_cb)

        self.TAB2 = Frame(self.notebook, bg="#00766c")
        self.notebook.add(self.TAB2, text='Play mode')
        self.notebook.pack(expand=2, fill="both")

        self.is_play_mode = 0
        self.is_recording = False

        # ─── RECORD MODE ─────────────────────────────────────────────────

        # record button.
        self.record_btn_text = StringVar(self.TAB1, name="record_btn_text")
        self.record_btn_text.set("Record")
        self.is_recording = False
        record_btn = Button(self.TAB1, textvariable=self.record_btn_text, command=self.record_btn_cb).grid(columnspan=2,
                                                                                                           sticky=W)

        # name box.
        self.name_box = Entry(self.TAB1, text="Sign name")
        self.name_box.grid(row=2, column=0, sticky=E)

        # num recoreds
        self.num_records_text = StringVar(self.TAB1, name="num_records_text")
        num_records_text_box = Label(self.TAB1, textvariable=self.num_records_text).grid(row=0, column=1)
        self.num_records_text.set("num records: 0")

        # save button.
        self.save_btn = Button(self.TAB1, text="Save", command=self.save_btn_cb)
        self.save_btn.grid(row=2, column=1, sticky=W)

        # ─── PLAY MODE ───────────────────────────────────────────────────

        # record button.
        self.record_btn_text = StringVar(self.TAB2, name="record_btn_text")
        self.record_btn_text.set("Record")
        self.is_recording = False
        record_btn_p = Button(self.TAB2, textvariable=self.record_btn_text,
                              command=self.record_btn_cb).grid(columnspan=2, sticky=W)

        # Show console.
        self.console_box = Text(self.TAB2, bg="#44a18e")
        self.console_box.grid(columnspan=2, sticky=W)

    def record_btn_cb(self):
        self.is_recording = not self.is_recording
        tab_id = self.notebook.index(self.notebook.select())
        if self.is_recording:
            self.record_btn_text.set("Stop")
            self.notebook.tab(not tab_id, state="disabled")
            self.name_box["state"] = DISABLED
            self.save_btn["state"] = DISABLED

        else:
            self.record_btn_text.set("Record")
            self.notebook.tab(not tab_id, state="normal")
            self.name_box["state"] = NORMAL
            self.save_btn["state"] = NORMAL

    def tab_btn_cb(self, event):
        self.is_play_mode = self.notebook.index("current")

    def save_btn_cb(self):
        pass


if __name__ == "__main__":
    config_dialog = DemoGUI()
    config_dialog.root.mainloop()
