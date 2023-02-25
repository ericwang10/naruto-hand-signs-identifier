import turtle

win = turtle.Screen()
win.title("Paddle")
win.bgcolor("black")
win.tracer(0)
win.setup(width = 600, height = 600)

while True:
    win.update()

paddle = turtle.Turtle()
paddle.shape("square")
paddle.shapesize(stretch_wid=1, stretch_len = 5)
paddle.color('white')
paddle.goto(0,-275)
paddle.speed(0)

#ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("red")
ball.penup()
ball.goto(0,100)
