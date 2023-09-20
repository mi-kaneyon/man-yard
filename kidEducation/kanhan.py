#Manyan3 presents 2023
# Revise the code to handle both character-to-reading and reading-to-character tasks
import pygame
import pandas as pd
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WIDTH, HEIGHT = 800, 600
MOVE_SPEED = 20
time_limit = 120
start_time = time.time()

# Load data
word_data = pd.read_excel("hanzi.xlsx").to_dict('records')

# Initialize screen
win = pygame.display.set_mode((WIDTH, HEIGHT))

# Initialize font
font = pygame.font.Font('Arial Unicode MS.TTF', 32)

# Function to get targets based on the player and player type
def get_targets(player, player_type, reading_type, word_data):
    correct = []
    incorrect = []
    
    for word in word_data:
        if player[player_type] == word.get(player_type, ''):
            correct.append(word)
        else:
            incorrect.append(word)

    correct_sample = random.sample(correct, min(1, len(correct)))
    incorrect_sample = random.sample(incorrect, min(3, len(incorrect)))

    targets = correct_sample + incorrect_sample
    random.shuffle(targets)
    return targets

# Initialize variables
player = random.choice(word_data)
player_type, reading_type = random.choice([("日本漢字", "日本よみ"), ("中文汉字", "中文拼音"), ("繁体中文", "中文注音")])
score = 0

# Initialize player and target positions
player_rect = pygame.Rect(WIDTH // 2, HEIGHT - 50, 50, 50)

targets = get_targets(player, player_type, reading_type, word_data)
target_rects = [pygame.Rect(random.randint(0, WIDTH-50), random.randint(0, HEIGHT-50), 50, 50) for _ in range(len(targets))]

# Main Loop
run = True
while run:
    elapsed_time = time.time() - start_time
    pygame.time.delay(100)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_rect.x -= MOVE_SPEED
    if keys[pygame.K_RIGHT]:
        player_rect.x += MOVE_SPEED
    if keys[pygame.K_UP]:
        player_rect.y -= MOVE_SPEED
    if keys[pygame.K_DOWN]:
        player_rect.y += MOVE_SPEED

    # Collision detection and scoring
    for i, target_rect in enumerate(target_rects):
        if player_rect.colliderect(target_rect):
            if targets[i][reading_type] == player[reading_type]:
                score += 1  # Increment the score for a correct answer
            else:
                score -= 1  # Decrement the score for a wrong answer

            # After updating the score, generate new player and targets
            player = random.choice(word_data)
            player_type, reading_type = random.choice([("日本漢字", "日本よみ"), ("中文汉字", "中文拼音"), ("繁体中文", "中文注音")])
            targets = get_targets(player, player_type, reading_type, word_data)
            target_rects = [pygame.Rect(random.randint(0, WIDTH-50), random.randint(0, HEIGHT-50), 50, 50) for _ in range(len(targets))]


    # Draw everything
    win.fill(WHITE)
    pygame.draw.rect(win, BLUE, player_rect)
    player_label = font.render(player[player_type], True, RED)
    win.blit(player_label, (player_rect.x, player_rect.y - 30))

    for i, target_rect in enumerate(target_rects):
        pygame.draw.rect(win, GREEN, target_rect)
        target_label = font.render(targets[i][reading_type], True, RED)
        win.blit(target_label, (target_rect.x, target_rect.y - 30))

    pygame.display.update()

    # Game Over Logic
    if elapsed_time > time_limit:
        win.fill(WHITE)
        game_over_text = f"Game Over! Your Score: {score}"
        text_surface = font.render(game_over_text, True, RED)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        win.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.delay(5000)
        run = False

pygame.quit()
