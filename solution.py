import numpy as np
import struct

class BallMovement():    
    
    def __init__(self):    
        self.pointBuffer = []
        self.fly = False
        self.minFlySpeed = 2
        self.maxFlySpeed = 20
        self.timeLim = 0.1
        self.polynom = False
        self.p = 1
        self.maxPredErr = 0.3
        self.mon = 0.5
        self.newTraectory = []
        self.polyCount = 10        
        self.kx = None
        self.ky = None
        self.kz = None
        self.count = 0
        self.pointReady = False
        
    def checkBeginVectorPoint(self, vector, point):
        v1 = np.array([
            (point.x - vector[1].x) / (point.t - vector[1].t),
            (point.y - vector[1].y) / (point.t - vector[1].t),
            (point.z - vector[1].z) / (point.t - vector[1].t),
        ])
        v2 = np.array([
            (vector[1].x - vector[0].x) / (vector[1].t - vector[0].t),
            (vector[1].y - vector[0].y) / (vector[1].t - vector[0].t),
            (vector[1].z - vector[0].z) / (vector[1].t - vector[0].t),
        ])
        print("Min fly speed v1 {0}".format(np.linalg.norm(v1)))
        print("Min fly speed v2 {0}".format(np.linalg.norm(v2)))
        print("Mono {0}".format(abs(np.linalg.norm(v1) / np.linalg.norm(v2) - 1)))

        print(np.linalg.norm(v1) > self.minFlySpeed)
        print(np.linalg.norm(v1) < self.maxFlySpeed)
        print(abs(np.linalg.norm(v1) / np.linalg.norm(v2) - 1) < self.mon)
        print((point.t - vector[0].t) < self.timeLim)
        print((v1[0]) < -self.minFlySpeed * 0.5)
        print((v2[0]) < -self.minFlySpeed * 0.5)
        print(point.x - vector[0].x < 0)

        if ((np.linalg.norm(v1) > self.minFlySpeed and np.linalg.norm(v1) < self.maxFlySpeed) 
            # and (np.linalg.norm(v1 - v2) < self.minFlySpeed * self.aspectRatio) 
            and abs(np.linalg.norm(v1) / np.linalg.norm(v2) - 1) < self.mon
            and ((point.t - vector[0].t) < self.timeLim) 
            and ((v1[0]) < -self.minFlySpeed * 0.5)
            and ((v2[0]) < -self.minFlySpeed * 0.5)
            and ((point.x - vector[0].x) < 0)):
            return True
        else:
            return False


    def checkBeginVector(self, vector):
        v1 = np.array([
            (vector[2].x - vector[1].x) / (vector[2].t - vector[1].t),
            (vector[2].y - vector[1].y) / (vector[2].t - vector[1].t),
            (vector[2].z - vector[1].z) / (vector[2].t - vector[1].t),
        ])
        v2 = np.array([
            (vector[1].x - vector[0].x) / (vector[1].t - vector[0].t),
            (vector[1].y - vector[0].y) / (vector[1].t - vector[0].t),
            (vector[1].z - vector[0].z) / (vector[1].t - vector[0].t),
        ])
        print("Min fly speed new traectory {0}".format(np.linalg.norm(v1)))
        if ((np.linalg.norm(v1) > self.minFlySpeed and np.linalg.norm(v1) < self.maxFlySpeed) 
            # and (np.linalg.norm(v1 - v2) < self.minFlySpeed * self.aspectRatio) 
            and abs(np.linalg.norm(v1) / np.linalg.norm(v2) - 1) < self.mon
            and ((vector[2].t - vector[0].t) < self.timeLim) 
            and ((v1[0]) < -self.minFlySpeed * 0.5)
            and ((v2[0]) < -self.minFlySpeed * 0.5)
            and ((vector[2].x - vector[0].x) < 0)):
            return True
        else:
            return False


    def calcX(self, t, Px, Vx, k):
        return Px + Vx * (1 - np.exp(-k * t)) / k

    def calcY(self, t, Py, Vy, k):
        return Py + Vy * (1 - np.exp(-k * t)) / k

    def calcZ(self, t, Pz, Vz, k, g):
        return Pz - (g * t + (np.exp(-k * t) - 1) * (g / k + Vz)) / k

    def calcVelX(self, t, Vx, k):
        return Vx * np.exp(-k * t)

    def calcVelY(self, t, Vy, k):
        return Vy * np.exp(-k * t)

    def calcVelZ(self, t, Vz, k, g):
        return -(g - np.exp(-k * t) * (g + k * Vz)) / k

    def calcT(self, Xref, Px, Vx, k):    
        return -np.log(1 - k * (Xref - Px) / Vx) / k


    def printBuffer(self):
        print("Buffer length {0}".format(len(self.pointBuffer)))
        for i in self.pointBuffer:
            print("t {} X {} Y {} Z {}".format(i.t, i.x, i.y, i.z))

    def pushPoint(self, point):        
        if (len(self.pointBuffer) < 2 and not self.fly):
            self.pointBuffer.append(point)
            print("11111")
        elif (len(self.pointBuffer) == 2 and not self.fly):
            print("22222")
            if (self.checkBeginVectorPoint(self.pointBuffer, point)):
                self.fly = True
            else:
                self.pointBuffer.pop(0)
                self.pointBuffer.append(point)
        if (self.fly):
            if (len(self.pointBuffer) < 3):
                self.pointBuffer.append(point)               
            elif self.polynom:

                ex = np.polyval(self.kx, point.t) - point.x
                ey = np.polyval(self.ky, point.t) - point.y
                ez = np.polyval(self.kz, point.t) - point.z
                print("ex {0}".format(ex))
                print("ey {0}".format(ey))
                print("ez {0}".format(ez))
                print("kx {0}".format(self.kx))
                print("ky {0}".format(self.ky))
                print("kz {0}".format(self.kz))
                print("Polyval {0}".format(np.polyval(self.kx, point.t)))
                print("Point {0}".format([point.x, point.y, point.z]))
                E = np.linalg.norm(np.array([ex, ey, ez]))                
                if (E > self.maxPredErr or not self.checkBeginVectorPoint(self.pointBuffer[-2:], point)):
                    self.newTraectory.append(point)
                else:
                    if (len(self.pointBuffer) == self.polyCount):
                        self.pointBuffer.pop(0)
                    self.pointBuffer.append(point)
                    self.newTraectory = []
                if (len(self.newTraectory) == 3):
                    self.pointBuffer = self.newTraectory
                    self.fly = False
                    self.polynom = False
                    self.newTraectory = []
                    if (self.checkBeginVector(self.pointBuffer)):
                        self.fly = True
                    else:
                        self.pointBuffer.pop(0)
            if (self.fly):
                self.printBuffer()
                x = np.array([p.x for p in self.pointBuffer])
                y = np.array([p.y for p in self.pointBuffer])
                z = np.array([p.z for p in self.pointBuffer])
                t = np.array([p.t for p in self.pointBuffer])
                if (len(self.pointBuffer) <= 3):
                    self.p = 1
                if (len(self.pointBuffer) > 3):
                    self.p = 2
                p = self.p
                self.kx = np.polyfit(t, x, p)
                self.ky = np.polyfit(t, y, p)
                self.kz = np.polyfit(t, z, p)
                dkx = np.polyder(self.kx)
                dky = np.polyder(self.ky)
                dkz = np.polyder(self.kz)
                self.polynom = True

                T = (self.pointBuffer[len(self.pointBuffer) - 1].t + self.pointBuffer[0].t) / 2
                PxT = np.polyval(self.kx,T)
                PyT = np.polyval(self.ky,T)
                PzT = np.polyval(self.kz,T)
                VxT = np.polyval(dkx, T)
                VyT = np.polyval(dky, T)
                VzT = np.polyval(dkz, T)
                # if (len(self.pointBuffer) == self.polyCount):
                print("PxT {}".format(PxT))
                print("PyT {}".format(PyT))
                print("PzT {}".format(PzT))
                print("VxT {}".format(VxT))
                print("VyT {}".format(VyT))
                print("VzT {}".format(VzT))
                t = self.calcT(0.60, PxT, VxT, 0.9)
                print("t {}".format(t))
                X = self.calcX(t, PxT, VxT, 0.9)
                Y = self.calcY(t, PyT, VyT, 0.9)
                Z = self.calcZ(t, PzT, VzT, 0.9, 9.8)
                velX = self.calcVelX(t, VxT, 0.9)
                velY = self.calcVelY(t, VyT, 0.9)
                velZ = self.calcVelZ(t, VzT, 0.9, 9.8)
                t = t - (self.pointBuffer[len(self.pointBuffer) - 1].t - T) 
                print("For KUKA time {0}".format(t))
                print("X  {}".format(X))
                print("Y  {}".format(Y))
                print("Z  {}".format(Z))
                print(velX)
                print(velY)
                print(velZ)
                return [t, X, Y, Z]
                self.count += 1

def calculate_kin(data, 
      inverse_kin_func, 
      forward_kin_func, 
      min_axes_func, 
      current_axes, 
      current_pos, 
      tool_offset, tool_angles, links, min_axes_limit, max_axes_limit):
      msg = struct.unpack("dddd", data)
      print("msg", msg)

      robot_position = np.array([msg[1], msg[2], msg[3]]) * 1000
      t = msg[0] - 0.2
      # t = 0.01
      # c = np.degrees(np.arctan2(robot_position[2] - centerZ, robot_position[1] - centerY))
      # c -= 90.0
      c = 0.0
      robot_angles = np.array([0.0, 0.0, c])
      # robot_position = np.array([msg[0], msg[1], msg[2]])
      # t = msg[0]
      print(robot_position)
      axes = np.zeros((8, 6))
      inverse_kin_func(robot_position, robot_angles, tool_offset, tool_angles, links, axes) #axes is result of inverse_kin_func

      # current_axes = np.zeros(6)
      # rc.get_axes(current_axes)

      target_axes = min_axes_func(axes, current_axes, min_axes_limit, max_axes_limit)
      target_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


      robot_position1 = robot_position + np.array([1.0, 0.0, 1.0]) 
      inverse_kin_func(robot_position1, robot_angles, tool_offset, tool_angles, links, axes)
      target_axes1 = min_axes_func(axes, current_axes, min_axes_limit, max_axes_limit)
      target_vel1 = (target_axes1 - target_axes) * 1000000000.0
      return (t, target_axes, target_vel1)
