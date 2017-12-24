

class Point:
    def __init__(self, x , y):
        self.x = x
        self.y = y


    def __str__(self):
        return "{0},{1}".format(self.x, self.y)


class Line:
    def __init__(self, p1 : Point , p2 : Point):
        self.p1 = p1
        self.p2 = p2
        self.threshold = 50

    def intersect(self, other):
        o1 = Helper.orientation(self.p1, self.p2, other.p1)
        o2 = Helper.orientation(self.p1, self.p2, other.p2)
        o3 = Helper.orientation(other.p1, other.p2, self.p1)
        o4 = Helper.orientation(other.p1, other.p2, self.p2)
        #general case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        if o1 == 0 and Helper.on_segment(self.p1, other.p1, self.p2):
            return True
        if o2 == 0 and Helper.on_segment(self.p1, other.p2, self.p2):
            return True
        if o3 == 0 and Helper.on_segment(other.p1, self.p1, other.p2):
            return True
        if o3 == 0 and Helper.on_segment(other.p1, self.p2, other.p2):
            return True
        return False


    def is_close_to(self, other):
        dis1 = abs(self.p1.x - other.p1.x)
        dis2 = abs(self.p2.x - other.p2.x)
        return max(dis1, dis2) < self.threshold

    def __str__(self):
        return "l: [{0}-{1}]".format(self.p1, self.p2)

class Helper:
    @staticmethod
    def orientation(p :Point, q: Point, r: Point):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if val == 0:
            return 0 # colinear
        elif val > 0:
            return 1
        return 2

    @staticmethod
    def on_segment(p: Point ,q: Point,r: Point):
        if q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y):
            return True
        return False;





