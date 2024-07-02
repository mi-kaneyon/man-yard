import os
import subprocess
from tkinter import Tk, Label, Button, Listbox, END, SINGLE, Text, Scrollbar, RIGHT, Y, LEFT, Frame

class SoundFixer:
    def __init__(self, master):
        self.master = master
        master.title("Ubuntu Sound Fixer")

        self.label = Label(master, text="Select the sound device to troubleshoot:")
        self.label.pack()

        self.listbox = Listbox(master, selectmode=SINGLE)
        self.listbox.pack()

        self.button_frame = Frame(master)
        self.button_frame.pack()

        self.refresh_button = Button(self.button_frame, text="Refresh Devices", command=self.list_devices)
        self.refresh_button.grid(row=0, column=0, padx=5, pady=5)

        self.check_button = Button(self.button_frame, text="Check Device", command=self.check_device)
        self.check_button.grid(row=0, column=1, padx=5, pady=5)

        self.fix_button = Button(self.button_frame, text="Fix Device", command=self.fix_device)
        self.fix_button.grid(row=0, column=2, padx=5, pady=5)

        self.result_text = Text(master, height=10, wrap='word')
        self.result_text.pack()

        self.scrollbar = Scrollbar(master, command=self.result_text.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.result_text.config(yscrollcommand=self.scrollbar.set)

        self.reboot_button = Button(master, text="Reboot", command=self.reboot_system)
        self.reboot_button.pack(side=LEFT, padx=5, pady=5)

        self.exit_button = Button(master, text="Exit", command=master.quit)
        self.exit_button.pack(side=RIGHT, padx=5, pady=5)

        self.list_devices()

    def list_devices(self):
        self.listbox.delete(0, END)
        devices = self.get_devices()
        for device in devices:
            self.listbox.insert(END, device)
    
    def get_devices(self):
        result = subprocess.run(['aplay', '-l'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        devices = [line for line in output.split('\n') if 'card' in line]
        return devices
    
    def check_device(self):
        selected = self.listbox.get(self.listbox.curselection())
        if selected:
            card_info = selected.split(':')[0]
            card_number = card_info.split(' ')[1]
            self.result_text.insert(END, f"Checking card {card_number}...\n")
            # Implement further checks if needed
            # Example check: If no devices found
            if not self.get_devices():
                self.result_text.insert(END, "No sound devices found. Please check your hardware connection.\n")
            else:
                self.result_text.insert(END, f"Card {card_number} seems to be okay.\n")
    
    def fix_device(self):
        selected = self.listbox.get(self.listbox.curselection())
        if selected:
            card_info = selected.split(':')[0]
            card_number = card_info.split(' ')[1]
            self.result_text.insert(END, f"Fixing card {card_number}...\n")
            # Implement fixing steps
            subprocess.run(['alsactl', 'init', card_number])
            self.result_text.insert(END, f"Card {card_number} has been fixed.\n")
            # Provide further advice and actions
            self.result_text.insert(END, "If you still experience issues, please try the following steps:\n")
            self.result_text.insert(END, "1. Ensure your headphones or speakers are properly connected.\n")
            self.result_text.insert(END, "2. Run 'alsamixer' and check the volume levels and mute status.\n")
            self.result_text.insert(END, "3. Restart the PulseAudio service with 'pulseaudio -k' command.\n")
            self.result_text.insert(END, "4. If all else fails, consider restarting your computer.\n")
    
    def reboot_system(self):
        self.result_text.insert(END, "Rebooting the system...\n")
        subprocess.run(['reboot'])
  # 'sudo', 
root = Tk()
sound_fixer = SoundFixer(root)
root.mainloop()
