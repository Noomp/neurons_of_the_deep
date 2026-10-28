"""Reference oracles transcribed from GLAM (flowersteam/Grounding_LLMs_with_online_RL).

* ``glam_gen_graph``: English branch of ``MiniGridEnv.gen_graph`` in
  babyai-text/gym-minigrid/gym_minigrid/minigrid.py, operating on the live env.
* ``glam_generate_prompt``: ``BaseAgent.generate_prompt`` in
  experiments/agents/base_agent.py.

Kept as close to the original as possible (only the French branches and
dead code were removed) so the tests check our port against the source.
"""

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX


def glam_gen_graph(env):
    self = env.unwrapped
    grid, vis_mask = self.gen_obs_grid()
    image = grid.encode(vis_mask)
    IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

    list_textual_descriptions = []
    if self.carrying is not None:
        list_textual_descriptions.append("You carry a {} {}".format(self.carrying.color, self.carrying.type))

    agent_pos_vx, agent_pos_vy = self.get_view_coords(self.agent_pos[0], self.agent_pos[1])

    view_field_dictionary = dict()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                if i not in view_field_dictionary.keys():
                    view_field_dictionary[i] = dict()
                    view_field_dictionary[i][j] = image[i][j]
                else:
                    view_field_dictionary[i][j] = image[i][j]

    # Find wall in front
    j = agent_pos_vy - 1
    object_seen = False
    while j >= 0 and not object_seen:
        if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
            if image[agent_pos_vx][j][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vy - j} step{'s' if agent_pos_vy - j > 1 else ''} forward"
                )
                object_seen = True
            else:
                object_seen = True
        j -= 1
    # Find wall left
    i = agent_pos_vx - 1
    object_seen = False
    while i >= 0 and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vx - i} step{'s' if agent_pos_vx - i > 1 else ''} left"
                )
                object_seen = True
            else:
                object_seen = True
        i -= 1
    # Find wall right
    i = agent_pos_vx + 1
    object_seen = False
    while i < image.shape[0] and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {i - agent_pos_vx} step{'s' if i - agent_pos_vx > 1 else ''} right"
                )
                object_seen = True
            else:
                object_seen = True
        i += 1

    for i in view_field_dictionary.keys():
        for j in view_field_dictionary[i].keys():
            if i != agent_pos_vx or j != agent_pos_vy:
                object = view_field_dictionary[i][j]
                relative_position = dict()
                if i - agent_pos_vx > 0:
                    relative_position["x_axis"] = ("right", i - agent_pos_vx)
                elif i - agent_pos_vx == 0:
                    relative_position["x_axis"] = ("face", 0)
                else:
                    relative_position["x_axis"] = ("left", agent_pos_vx - i)
                if agent_pos_vy - j > 0:
                    relative_position["y_axis"] = ("forward", agent_pos_vy - j)
                elif agent_pos_vy - j == 0:
                    relative_position["y_axis"] = ("forward", 0)

                distances = []
                if relative_position["x_axis"][0] in ["face", "en face"]:
                    distances.append((relative_position["y_axis"][1], relative_position["y_axis"][0]))
                elif relative_position["y_axis"][1] == 0:
                    distances.append((relative_position["x_axis"][1], relative_position["x_axis"][0]))
                else:
                    distances.append((relative_position["x_axis"][1], relative_position["x_axis"][0]))
                    distances.append((relative_position["y_axis"][1], relative_position["y_axis"][0]))

                description = ""
                if object[0] != 4:  # if it is not a door
                    description = f"You see a {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                else:
                    description = f"You see a {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "

                for _i, _distance in enumerate(distances):
                    if _i > 0:
                        description += " and "
                    description += f"{_distance[0]} step{'s' if _distance[0] > 1 else ''} {_distance[1]}"

                list_textual_descriptions.append(description)

    return {"descriptions": list_textual_descriptions}


def glam_generate_prompt(goal, subgoals, deque_obs, deque_actions):
    ldo = len(deque_obs)
    lda = len(deque_actions)

    head_prompt = "Possible action of the agent:"
    for sg in subgoals:
        head_prompt += " {},".format(sg)
    head_prompt = head_prompt[:-1]

    g = " \n Goal of the agent: {}".format(goal)
    obs = ""
    for i in range(ldo):
        obs += " \n Observation {}: ".format(i)
        for d_obs in deque_obs[i]:
            obs += "{}, ".format(d_obs)
        obs += "\n Action {}: ".format(i)
        if i < lda:
            obs += "{}".format(deque_actions[i])
    return head_prompt + g + obs
