from itertools import cycle
import random
import sys
import os
import pygame
from pygame.locals import *
import numpy as np

from Genetics.populationclass import Population
from Genetics.birdnetclass import BirdNet

# Display settings
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 130  # Vertical gap between upper and lower pipe
PIPE_SPACING = 200  # Horizontal distance between pipe pairs
BASEY = SCREENHEIGHT * 0.79

# Training settings
RENDER_EVERY = 5

# Physics - tuned for proper flappy feel
GRAVITY = 0.5
FLAP_STRENGTH = -8
MAX_FALL_SPEED = 8
PIPE_SPEED = -3

# Game states
STATE_MENU = 'menu'
STATE_MODEL_SELECT = 'model_select'
STATE_PLAY = 'play'
STATE_RUN_MODEL = 'run_model'
STATE_TRAIN = 'train'

# Assets
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
address_assets = 'assets/sprites/'
models_dir = 'models/'

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

PLAYERS_LIST = (
    (address_assets + 'redbird-upflap.png', address_assets + 'redbird-midflap.png', address_assets + 'redbird-downflap.png'),
    (address_assets + 'bluebird-upflap.png', address_assets + 'bluebird-midflap.png', address_assets + 'bluebird-downflap.png'),
    (address_assets + 'yellowbird-upflap.png', address_assets + 'yellowbird-midflap.png', address_assets + 'yellowbird-downflap.png'),
)
BACKGROUNDS_LIST = (address_assets + 'background-day.png', address_assets + 'background-night.png')
PIPES_LIST = (address_assets + 'pipe-green.png', address_assets + 'pipe-red.png')


# Cache for scaled surface
_scale_cache = {'size': None, 'surface': None}

def scale_to_screen(surface, screen):
    """Scale game surface to fit current window size, with caching."""
    current_size = screen.get_size()

    # Only create new scaled surface if window size changed
    if _scale_cache['size'] != current_size:
        _scale_cache['size'] = current_size
        _scale_cache['surface'] = pygame.Surface(current_size)

    # Scale directly into cached surface
    pygame.transform.scale(surface, current_size, _scale_cache['surface'])
    return _scale_cache['surface']


def get_saved_models():
    """Return list of saved model files."""
    if not os.path.exists(models_dir):
        return []
    return [f[:-4] for f in os.listdir(models_dir) if f.endswith('.npz')]


def save_model(bird, name):
    """Save a bird's weights to file."""
    filepath = os.path.join(models_dir, f"{name}.npz")
    np.savez(filepath, *bird.tensors)
    print(f"Model saved to {filepath}")


def load_model(name):
    """Load weights into a new BirdNet."""
    filepath = os.path.join(models_dir, f"{name}.npz")
    bird = BirdNet()
    data = np.load(filepath)
    bird.tensors = [data[f'arr_{i}'] for i in range(len(data.files))]
    return bird


def load_assets():
    """Load all game assets."""
    # Number sprites
    IMAGES['numbers'] = tuple(
        pygame.image.load(address_assets + f'{i}.png').convert_alpha()
        for i in range(10)
    )
    IMAGES['message'] = pygame.image.load(address_assets + 'message.png').convert_alpha()
    IMAGES['base'] = pygame.image.load(address_assets + 'base.png').convert_alpha()
    IMAGES['dot'] = pygame.image.load(address_assets + 'dot.png').convert_alpha()
    IMAGES['bird_dot'] = pygame.image.load(address_assets + 'bird_dot.png').convert_alpha()
    IMAGES['gameover'] = pygame.image.load(address_assets + 'gameover.png').convert_alpha()


def load_random_sprites():
    """Load random background, pipe, and player sprites."""
    IMAGES['background'] = pygame.image.load(random.choice(BACKGROUNDS_LIST)).convert()
    pipe_img = pygame.image.load(random.choice(PIPES_LIST)).convert_alpha()
    IMAGES['pipe'] = (pygame.transform.rotate(pipe_img, 180), pipe_img)
    player_set = random.choice(PLAYERS_LIST)
    IMAGES['player'] = tuple(pygame.image.load(p).convert_alpha() for p in player_set)

    HITMASKS['pipe'] = (getHitmask(IMAGES['pipe'][0]), getHitmask(IMAGES['pipe'][1]))
    HITMASKS['player'] = tuple(getHitmask(img) for img in IMAGES['player'])


def draw_text(surface, text, x, y, font, color=(255, 255, 255), center=False):
    """Draw text on surface."""
    rendered = font.render(text, True, color)
    if center:
        x = x - rendered.get_width() // 2
    surface.blit(rendered, (x, y))


def main_menu(screen, game_surface, font, font_large):
    """Main menu screen."""
    options = ['PLAY GAME', 'RUN MODEL', 'TRAIN', 'EXIT']
    selected = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == K_RETURN:
                    if options[selected] == 'PLAY GAME':
                        return STATE_PLAY, None
                    elif options[selected] == 'RUN MODEL':
                        return STATE_MODEL_SELECT, 'run'
                    elif options[selected] == 'TRAIN':
                        return STATE_MODEL_SELECT, 'train'
                    elif options[selected] == 'EXIT':
                        pygame.quit()
                        sys.exit()
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Draw menu
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        # Title
        draw_text(game_surface, 'FLAPPY BIRD', SCREENWIDTH // 2, 80, font_large, (255, 255, 255), center=True)
        draw_text(game_surface, 'ML', SCREENWIDTH // 2, 120, font_large, (255, 200, 0), center=True)

        # Menu options
        for i, option in enumerate(options):
            color = (255, 255, 0) if i == selected else (255, 255, 255)
            prefix = '> ' if i == selected else '  '
            draw_text(game_surface, prefix + option, SCREENWIDTH // 2, 200 + i * 40, font, color, center=True)

        # Scale and display
        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def model_select_menu(screen, game_surface, font, font_large, mode):
    """Model selection submenu."""
    models = get_saved_models()
    options = ['NEW (random weights)'] + models + ['BACK']
    selected = 0
    clock = pygame.time.Clock()

    title = 'SELECT MODEL TO RUN' if mode == 'run' else 'SELECT STARTING WEIGHTS'

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == K_RETURN:
                    if options[selected] == 'BACK':
                        return STATE_MENU, None
                    elif options[selected] == 'NEW (random weights)':
                        if mode == 'run':
                            return STATE_RUN_MODEL, None
                        else:
                            return STATE_TRAIN, None
                    else:
                        if mode == 'run':
                            return STATE_RUN_MODEL, options[selected]
                        else:
                            return STATE_TRAIN, options[selected]
                elif event.key == K_ESCAPE:
                    return STATE_MENU, None

        # Draw
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        draw_text(game_surface, title, SCREENWIDTH // 2, 60, font, (255, 255, 255), center=True)

        # Scrollable list (show max 8 items)
        start_idx = max(0, selected - 4)
        end_idx = min(len(options), start_idx + 8)

        for i, idx in enumerate(range(start_idx, end_idx)):
            option = options[idx]
            color = (255, 255, 0) if idx == selected else (255, 255, 255)
            prefix = '> ' if idx == selected else '  '
            draw_text(game_surface, prefix + option[:20], SCREENWIDTH // 2, 120 + i * 35, font, color, center=True)

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def play_game(screen, game_surface, font):
    """Manual play mode - single player controlled bird."""
    load_random_sprites()

    bird_x = SCREENWIDTH * 0.2
    bird_y = SCREENHEIGHT / 2
    bird_vel_y = 0

    basex = 0
    baseShift = IMAGES['base'].get_width() - SCREENWIDTH

    # Pipes
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
    ]

    score = 0
    clock = pygame.time.Clock()

    while True:
        flap = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return STATE_MENU
                if event.key == K_SPACE or event.key == K_UP:
                    flap = True

        # Flap
        if flap and bird_y > 0:
            bird_vel_y = FLAP_STRENGTH

        # Gravity
        if bird_vel_y < MAX_FALL_SPEED:
            bird_vel_y += GRAVITY
        bird_y += bird_vel_y

        # Move pipes
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += PIPE_SPEED
            lPipe['x'] += PIPE_SPEED

        # Add new pipe when last pipe has moved onto screen enough
        if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # Remove off-screen pipe
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # Check score
        playerMidPos = bird_x + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1

        # Check crash
        crash = checkCrash({'x': bird_x, 'y': bird_y, 'index': 0}, upperPipes, lowerPipes)
        if crash[0]:
            return STATE_MENU

        # Draw
        basex = -((-basex + 4) % baseShift)
        game_surface.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        game_surface.blit(IMAGES['base'], (basex, BASEY))
        game_surface.blit(IMAGES['player'][1], (bird_x, bird_y))
        showScore(game_surface, score)

        draw_text(game_surface, 'ESC: Menu', 5, 5, font, (255, 255, 255))

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def run_model_game(screen, game_surface, font, model_name):
    """Run a trained model - watch single AI bird play."""
    load_random_sprites()

    # Load or create bird
    if model_name:
        bird = load_model(model_name)
    else:
        bird = BirdNet()

    bird.x = SCREENWIDTH * 0.2
    bird.y = SCREENHEIGHT / 2
    bird.birdVelY = 0

    basex = 0
    baseShift = IMAGES['base'].get_width() - SCREENWIDTH

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
    ]

    closest_pipe_x = upperPipes[0]['x']
    closest_pipe_y = lowerPipes[0]['y']

    score = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                return STATE_MENU

        # AI decision
        bird.set_input((closest_pipe_x + 20) - (bird.x + 12), (bird.y + 12) - (closest_pipe_y - 50))
        if bird.fly_up() and bird.y > 0:
            bird.birdVelY = FLAP_STRENGTH

        # Physics
        if bird.birdVelY < MAX_FALL_SPEED:
            bird.birdVelY += GRAVITY
        bird.y += bird.birdVelY

        # Move pipes
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += PIPE_SPEED
            lPipe['x'] += PIPE_SPEED
        closest_pipe_x += PIPE_SPEED

        if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)
            closest_pipe_x = upperPipes[0]['x']
            closest_pipe_y = lowerPipes[0]['y']

        # Score
        playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if len(upperPipes) > 1:
                    closest_pipe_x = upperPipes[1]['x']
                    closest_pipe_y = lowerPipes[1]['y']

        # Crash
        crash = checkCrash({'x': bird.x, 'y': bird.y, 'index': 0}, upperPipes, lowerPipes)
        if crash[0]:
            return STATE_MENU

        # Draw
        basex = -((-basex + 4) % baseShift)
        game_surface.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        game_surface.blit(IMAGES['base'], (basex, BASEY))
        game_surface.blit(IMAGES['player'][1], (bird.x, bird.y))
        showScore(game_surface, score)

        name_display = model_name if model_name else 'Random'
        draw_text(game_surface, f'Model: {name_display}', 5, 5, font, (255, 255, 255))
        draw_text(game_surface, 'ESC: Menu', 5, 25, font, (255, 255, 255))

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def train_game(screen, game_surface, font, model_name):
    """Training mode with evolution."""
    load_random_sprites()

    # Initialize population
    population = Population()
    if model_name:
        # Load weights into all birds
        template = load_model(model_name)
        for bird in population.individuals:
            for i, tensor in enumerate(template.tensors):
                bird.tensors[i] = tensor.copy()
            bird.mutate()  # Add some variation

    birds = population.individuals
    generation = 1
    best_score_ever = 0

    clock = pygame.time.Clock()

    while True:  # Generation loop
        # Reset birds
        for bird in birds:
            bird.x = SCREENWIDTH * 0.2
            bird.y = SCREENHEIGHT / 2
            bird.birdVelY = 0
            bird.distance = 0
            bird.score = 0
            bird.birdFlapped = False

        crashTest = [[False, False] for _ in birds]

        basex = 0
        baseShift = IMAGES['base'].get_width() - SCREENWIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        upperPipes = [
            {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
        ]
        lowerPipes = [
            {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
        ]

        closest_pipe_x = upperPipes[0]['x']
        closest_pipe_y = lowerPipes[0]['y']
        score = 0
        frame_count = 0

        # Game loop for this generation
        while True:
            frame_count += 1
            save_and_exit = False

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    save_and_exit = True

            if save_and_exit:
                # Save best bird
                population.sort()
                best_bird = population.individuals[0]
                save_name = f"gen{generation}_score{best_score_ever}"
                save_model(best_bird, save_name)
                return STATE_MENU

            # Bird AI decisions
            for bird in birds:
                bird.set_input((closest_pipe_x + 20) - (bird.x + 12), (bird.y + 12) - (closest_pipe_y - 50))
                if bird.fly_up() and bird.y > 0:
                    bird.birdVelY = FLAP_STRENGTH
                    bird.birdFlapped = True

            # Check crashes
            for i, bird in enumerate(birds):
                if not crashTest[i][0]:
                    crashTest[i] = checkCrash({'x': bird.x, 'y': bird.y, 'index': 0}, upperPipes, lowerPipes)

            # Check if all dead
            all_dead = all(ct[0] for ct in crashTest)
            if all_dead:
                best_score_ever = max(best_score_ever, score)
                print(f"Gen {generation} | Score: {score} | Best: {best_score_ever}")
                population.evolve()
                generation += 1
                break

            # Update birds
            for i, bird in enumerate(birds):
                if not crashTest[i][0]:
                    bird.distance += abs(PIPE_SPEED)
                    if bird.birdVelY < MAX_FALL_SPEED and not bird.birdFlapped:
                        bird.birdVelY += GRAVITY
                    if bird.birdFlapped:
                        bird.birdFlapped = False
                    bird.y += bird.birdVelY

            # Move pipes
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                uPipe['x'] += PIPE_SPEED
                lPipe['x'] += PIPE_SPEED
            closest_pipe_x += PIPE_SPEED

            if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
                newPipe = getRandomPipe()
                upperPipes.append(newPipe[0])
                lowerPipes.append(newPipe[1])

            if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
                upperPipes.pop(0)
                lowerPipes.pop(0)
                closest_pipe_x = upperPipes[0]['x']
                closest_pipe_y = lowerPipes[0]['y']

            # Score check
            for i, bird in enumerate(birds):
                if not crashTest[i][0]:
                    playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
                    for j, pipe in enumerate(upperPipes):
                        pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                        if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                            bird.score += 1
                            score = max(score, bird.score)
                            if len(upperPipes) > 1:
                                closest_pipe_x = upperPipes[1]['x']
                                closest_pipe_y = lowerPipes[1]['y']

            # Render every N frames
            if frame_count % RENDER_EVERY == 0:
                basex = -((-basex + 20) % baseShift)
                game_surface.blit(IMAGES['background'], (0, 0))

                for uPipe, lPipe in zip(upperPipes, lowerPipes):
                    game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                    game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

                game_surface.blit(IMAGES['base'], (basex, BASEY))

                # Draw alive birds
                for i, bird in enumerate(birds):
                    if not crashTest[i][0]:
                        game_surface.blit(IMAGES['player'][1], (bird.x, bird.y))

                showScore(game_surface, score)

                # Stats
                alive = sum(1 for ct in crashTest if not ct[0])
                draw_text(game_surface, f'Gen: {generation}', 5, 5, font, (255, 255, 255))
                draw_text(game_surface, f'Alive: {alive}/{len(birds)}', 5, 25, font, (255, 255, 255))
                draw_text(game_surface, f'Best: {best_score_ever}', 5, 45, font, (255, 255, 255))
                draw_text(game_surface, 'ESC: Save & Exit', 5, SCREENHEIGHT - 20, font, (255, 255, 0))

                screen.blit(scale_to_screen(game_surface, screen), (0, 0))
                pygame.display.update()
                clock.tick(60)


def getRandomPipe():
    """Returns a randomly generated pipe."""
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def showScore(surface, score):
    """Display score on surface."""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = sum(IMAGES['numbers'][d].get_width() for d in scoreDigits)
    Xoffset = (SCREENWIDTH - totalWidth) / 2
    for digit in scoreDigits:
        surface.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """Returns True if player collides with base or pipes."""
    pw = IMAGES['player'][0].get_width()
    ph = IMAGES['player'][0].get_height()

    # Ground/ceiling collision
    if player['y'] + ph >= BASEY - 1 or player['y'] < 0:
        return [True, True]

    # Shrink hitbox slightly for more forgiving collisions
    margin = 3
    playerRect = pygame.Rect(player['x'] + margin, player['y'] + margin,
                              pw - margin * 2, ph - margin * 2)

    pipeW = IMAGES['pipe'][0].get_width()
    pipeH = IMAGES['pipe'][0].get_height()

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

        if playerRect.colliderect(uPipeRect) or playerRect.colliderect(lPipeRect):
            return [True, False]

    return [False, False]


def getHitmask(image):
    """Returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def main():
    pygame.init()

    # Resizable window, starts at 2x size, with double buffering
    screen = pygame.display.set_mode((SCREENWIDTH * 2, SCREENHEIGHT * 2), pygame.RESIZABLE | pygame.DOUBLEBUF)
    # Fixed-size render surface (game logic always uses this resolution)
    game_surface = pygame.Surface((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird ML')

    # Fonts
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)

    # Load assets
    load_assets()
    load_random_sprites()

    # Game state machine
    state = STATE_MENU
    model_name = None
    pending_mode = None

    while True:
        if state == STATE_MENU:
            state, pending_mode = main_menu(screen, game_surface, font, font_large)

        elif state == STATE_MODEL_SELECT:
            state, model_name = model_select_menu(screen, game_surface, font, font_large, pending_mode)
            if state == STATE_MENU:
                model_name = None

        elif state == STATE_PLAY:
            state = play_game(screen, game_surface, font)
            load_random_sprites()

        elif state == STATE_RUN_MODEL:
            state = run_model_game(screen, game_surface, font, model_name)
            model_name = None
            load_random_sprites()

        elif state == STATE_TRAIN:
            state = train_game(screen, game_surface, font, model_name)
            model_name = None
            load_random_sprites()


if __name__ == '__main__':
    main()
