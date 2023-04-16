import random
import threading
import tkinter as tk



class Dice:
    def __init__(self,dice_side,dice_no=1):
        self.dice_side = dice_side
        self.dice_no = dice_no

    def roll(self):
        return ([random.randint(1, self.dice_side+1) for i in range(1, self.dice_no+1)])

    def __str__(self):
        return f"This roll has total {self.dice_no} dice(s) with {self.dice_side} sides."



##### MAIN #####
def main():

    def rollDice_and_print():

        # ds = int(set_dice_side_entery.get())
        # dn = int(set_dice_no_entery.get())

        try:
            ds = int(set_dice_side_entery.get())
            dn = int(set_dice_no_entery.get())
        except ValueError:
            # handle the error when the input is not a valid integer
            result_text_box.insert(tk.INSERT, '[ERROR]***Input must be a positive number!*** \n', "error")
            result_text_box.see(tk.END)
            
            #Configure the text widget with certain color
            result_text_box.tag_config("error", foreground="red")


        #nth_dice = Dice(random.randint(1,6),random.randint(1,10))
        nth_dice = Dice(ds,dn)
        
        result_text_box.insert(tk.INSERT, nth_dice)
        result_text_box.insert(tk.INSERT, '\n')
        result_text_box.insert(tk.INSERT, nth_dice.roll())
        result_text_box.insert(tk.INSERT, '\n')
        result_text_box.see(tk.END)

    def rest_dafule_dsdn(ds,dn):
        set_dice_side_entery.delete(0, tk.END)
        set_dice_no_entery.delete(0, tk.END)

        set_dice_side_entery.insert(0, ds)
        set_dice_no_entery.insert(0, dn)

    def clear_text():
        result_text_box.delete('1.0', tk.END)

    def close_app():
        root.destroy()
    
    
    # Default Variable
    default_dice_side=6
    default_dice_no=1
    
    
    ##### main loop of tk GUI #####
    root = tk.Tk()
    #root.geometry("400x200")
    root.title('Dice App')
    root.resizable(0, 0)
     
    
    set_dice_side_labe1 = tk.Label(root, text="Set Dice Side")
    set_dice_side_labe1.grid(column=0, row=2, sticky=tk.W, padx=5, pady=0, ipadx=0, ipady=0)

    set_dice_no_labe1 = tk.Label(root, text="Set No# of Dices")
    set_dice_no_labe1.grid(column=0, row=3, sticky=tk.W, padx=5, pady=0, ipadx=0, ipady=0)

    set_dice_side_entery = tk.Entry(root)
    set_dice_side_entery.insert(0, default_dice_side)
    set_dice_side_entery.grid(column=1, row=2, sticky=tk.EW, padx=0, pady=0)

    set_dice_no_entery = tk.Entry(root)
    set_dice_no_entery.insert(0, default_dice_no)
    set_dice_no_entery.grid(column=1, row=3, sticky=tk.EW, padx=0, pady=0)

    set_default_button = tk.Button(root, text="Set Default Value", command=lambda: rest_dafule_dsdn(default_dice_side, default_dice_no))
    set_default_button.grid(column=2, row=2, sticky=tk.EW, padx=5, pady=5, ipadx=0, ipady=0)

    result_text_box_scbar = tk.Scrollbar(root)
    result_text_box_scbar.grid(column=3, row=6, sticky=tk.NS)
    
    result_text_box = tk.Text(root, height=10, width=50)
    result_text_box.grid(column=0, row=6, rowspan=1, columnspan=3, sticky=tk.W, padx=5, pady=5, ipadx=0, ipady=0)

    result_text_box_scbar.config(command=result_text_box.yview)
    result_text_box.config(yscrollcommand=result_text_box_scbar.set)

    roll_button = tk.Button(root, text="Roll Dice", command=rollDice_and_print)
    roll_button.grid(column=0, row=4, sticky=tk.EW, padx=5, pady=5, ipadx=0, ipady=0)
    
    clear_button = tk.Button(root, text="Clear Text Box", command=clear_text)
    clear_button.grid(column=1, row=4, sticky=tk.EW, padx=5, pady=5, ipadx=0, ipady=0)

    close_button = tk.Button(root, text="Close App", command=close_app)
    close_button.grid(column=2, row=4, sticky=tk.EW, padx=5, pady=5, ipadx=0, ipady=0)


    root.mainloop()





if __name__ == "__main__":
    main()
