# startup shell script
- It is used for my work to execute terminal commands when start-up Linux OS(Ubuntu)

  1. you create startup Applications Preferences from Ubuntu menu
  2. Add you sh script temporary
  3. Open setting file on the terminal
 
  ```
  #example
[Desktop Entry]
Type=Application
Exec=gnome-terminal -- bash -c '/home/educate/startdemo.sh; exec bash'
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Start StableDiffusion
Comment=Start the StableDiffusion WebUI
Terminal=false

 ```



# set your shell script proper dierectory


 
