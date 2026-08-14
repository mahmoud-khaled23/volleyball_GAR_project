# Box info for every single player
class BoxInfo:
    def __init__(self, line):
        words = line.split()

        self.category = words.pop()
        words = [int(word) for word in words]

        player_id, x1, y1, x2, y2, frame_id, lost, grouping, generated = words
        self.player_id = player_id
        self.box = [x1, y1, x2, y2]
        self.frame_id = frame_id
        self.lost = lost
        self.grouping = grouping
        self.generated = generated

    def get_frame_id(self):
        return self.frame_id

    def get_lost(self):
        return self.lost

    def get_box_info(self):
        return {'frame_id': self.frame_id,
                'box': self.box,
                'category': self.category
                }


