import random
import threading




class Dice:
    def __init__(self,dice_side,dice_no=1):
        self.dice_side = dice_side
        self.dice_no = dice_no

    def roll(self):
        return ([random.randint(1, self.dice_side+1) for i in range(1, self.dice_no+1)])

    def __str__(self):
        return f"This object has total of {self.dice_no} dice(s) with {self.dice_side} sides."

##### MAIN #####
#num = int(input("Enter a non-negative integer: "))
nth_dice = Dice(random.randint(1,6),random.randint(1,10))



# Create a new thread
t = []

for i in range(random.randint(1,9)):
  t.append(threading.Thread(target=(print (nth_dice.roll()))))

# Start the thread
for threads in t:
  threads.start()

# Wait for the thread to finish
# threads.join()
