import sys
import json
import socket
import math
import time
import os
import logging
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QSurfaceFormat, QFont, QColor, QPainter, QIcon, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QGridLayout, QSlider,
    QTextEdit, QPushButton, QHeaderView, QSpacerItem, QSizePolicy, QFrame,
    QScrollArea
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import pyqtgraph.opengl as gl  # Ensure pyqtgraph.opengl is imported
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
from PIL import Image

# ================================
# Logging Configuration
# ================================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# ================================
# Network Communication
# ================================

class FrontendClient:
    """
    Frontend client class that communicates with the backend via socket.
    Assumes that each JSON message from the backend ends with a newline character '\n'.
    """
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.ally_positions = []
        self.enemy_positions = []
        self.enemy_groups = []  # 新增属性，用于存储敌方组别及其威胁值
        self.human_preference = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # 人类设置偏好
        self.agent_preference = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # 智能体训练得到的偏好
        self.mixed_priority = [[1, 2, 3, 4, 5, 6]]

        self.connect_server()

    def connect_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            self.connected = True
            self.sock.settimeout(None)
            logging.info("Connected to backend")
        except Exception as e:
            logging.error(f"Failed to connect to backend: {e}")
            self.connected = False

    def close(self):
        if self.sock:
            self.sock.close()
        self.connected = False
        logging.info("Disconnected from backend")

    def send_preference(self, new_pref):
        if not self.connected:
            logging.warning("Not connected to backend. Cannot send preferences.")
            return
        data = {
            "type": "update_preference",
            "data": new_pref
        }
        msg = json.dumps(data) + "\n"  # Each JSON message ends with a newline
        try:
            self.sock.sendall(msg.encode('utf-8'))
            logging.info(f"Sent preferences: {new_pref}")
        except Exception as e:
            logging.error(f"Error sending preferences: {e}")
            self.connected = False

    def update_positions(self, info):
        self.ally_positions = info.get("ally", [])
        self.enemy_positions = info.get("enemy", [])
        self.enemy_groups = info.get("enemy_groups", [])  # 更新敌方组别信息
        self.agent_preference = info.get("agent_preference", self.agent_preference)
        self.mixed_priority = info.get("mixed_priority", self.mixed_priority)


class DataListenerThread(QThread):
    """
    A separate thread that continuously receives data using blocking recv from the socket.
    Each data message ends with a newline character, so the buffer is split by lines and JSON is parsed line by line.
    When data is received, it emits the data_received signal to send data to the main thread.
    """
    data_received = pyqtSignal(dict)

    def __init__(self, client: FrontendClient, parent=None):
        super().__init__(parent)
        self.client = client
        self.running = True
        self.buffer = ""

    def run(self):
        if not self.client.connected or self.client.sock is None:
            logging.warning("Not connected to backend. Data listener thread terminated.")
            return
        sock = self.client.sock
        sock.setblocking(True)  # Blocking mode
        logging.info("Data listener thread started.")
        while self.running:
            try:
                data = sock.recv(4096)
                if not data:
                    logging.warning("Backend has disconnected")
                    self.running = False
                    break
                self.buffer += data.decode('utf-8')
                self.parse_buffer_by_line()
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                self.running = False
                break

    def parse_buffer_by_line(self):
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]  # Keep the last incomplete line
        lines = lines[:-1]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
                self.data_received.emit(info)
                # logging.debug(f"Received data: {info}")
            except json.JSONDecodeError as ex:
                logging.error(f"JSON parsing failed: {ex}")

    def stop(self):
        self.running = False
        if self.client.connected:
            self.client.close()
        logging.info("Data listener thread stopped.")


# ================================
# 3D Rendering
# ================================

class Plane3D:
    """
    Represents a 3D airplane model, including model, texture, etc.
    Uses trimesh to load the model and convert it into OpenGL-renderable data.
    """
    def __init__(self, model_path, color=QColor("blue")):
        self.scene = None
        self.color = color
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.indices = []
        self.texture_id = None
        try:
            # Use absolute path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_model_path = os.path.join(current_dir, model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"Model file not found: {full_model_path}")
            self.load_model(full_model_path)
            logging.info(f"Successfully loaded model: {full_model_path}")
        except Exception as e:
            logging.error(f"Failed to load model ({model_path}): {e}")

    def load_model(self, model_path):
        mesh = trimesh.load(model_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Cannot load model: {model_path}")

        # Get vertices, normals, and texture coordinates
        self.vertices = mesh.vertices.astype(np.float32)
        self.normals = mesh.vertex_normals.astype(np.float32) if mesh.vertex_normals is not None else None

        # Check for texture coordinates
        if mesh.visual.kind == 'texture' and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            self.texcoords = mesh.visual.uv.astype(np.float32)
        else:
            self.texcoords = None  # No texture coordinates

        self.indices = mesh.faces.flatten().astype(np.uint32)

        # Load texture
        if mesh.visual.kind == 'texture' and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            image = mesh.visual.material.image
            image = Image.fromarray(image)
            self.texture_id = self.load_texture(image)

    def load_texture(self, image):
        try:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()
            width, height = image.size

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            logging.info(f"Texture loaded successfully: {image}")
            return texture_id
        except Exception as e:
            logging.error(f"Failed to load texture: {e}")
            return None

    def setup_buffers(self):
        """
        Set up VBOs
        """
        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        if self.normals is not None:
            self.vbo_normals = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        else:
            self.vbo_normals = None

        if self.texcoords is not None:
            self.vbo_texcoords = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_texcoords)
            glBufferData(GL_ARRAY_BUFFER, self.texcoords.nbytes, self.texcoords, GL_STATIC_DRAW)
        else:
            self.vbo_texcoords = None

        self.vbo_indices = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def render(self):
        try:
            glPushMatrix()  # Start applying transformations
            glRotatef(180, 0, 1, 0)  # Rotate 180 degrees to align +Z if model defaults to -Z

            if not hasattr(self, 'vbo_vertices'):
                self.setup_buffers()

            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glVertexPointer(3, GL_FLOAT, 0, None)

            if self.normals is not None and hasattr(self, 'vbo_normals'):
                glEnableClientState(GL_NORMAL_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
                glNormalPointer(GL_FLOAT, 0, None)
            else:
                glDisableClientState(GL_NORMAL_ARRAY)

            if self.texcoords is not None and hasattr(self, 'vbo_texcoords'):
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_texcoords)
                glTexCoordPointer(2, GL_FLOAT, 0, None)
            else:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)

            if self.texture_id and self.texcoords is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.texture_id)
            else:
                glDisable(GL_TEXTURE_2D)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

            glDisableClientState(GL_VERTEX_ARRAY)
            if self.normals is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
            if self.texcoords is not None:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)

            if self.texture_id and self.texcoords is not None:
                glDisable(GL_TEXTURE_2D)

        except Exception as e:
            logging.error(f"Exception during model rendering: {e}")
        finally:
            glPopMatrix()  # Ensure the stack is correctly released regardless of exceptions


class Plane3DInstance:
    """
    Manages a single airplane instance, including position, rotation, color, and speed.
    """
    def __init__(self, model: Plane3D, position, rotation, color=QColor("blue"), group=None, is_enemy=False):
        self.model = model  # Plane3D object
        self.position = position  # [x, y, z]
        self.rotation = rotation  # {"pitch":..., "yaw":..., "roll":...}
        self.color = color
        self.scale = 40  # Added scaling factor
        self.group = group  # Only enemies have group labels
        self.is_enemy = is_enemy  # Flag to indicate if it's an enemy plane
        self.command = [-1, -1]  # Initialize commands as [-1, -1] indicating no command
        self.speed = 0.0  # 新增属性，用于存储飞机速度

    def update_transform(self, position, rotation, color, group=None, command=None, speed=0.0):
        self.position = position
        self.rotation = rotation
        self.color = color
        if group is not None:
            self.group = group

        if command:
            self.command = command  # Update command information

        self.speed = speed  # 更新速度

    def render(self):
        try:
            glPushMatrix()
            glTranslatef(*self.position)

            # Apply rotations: yaw first, then pitch, then roll
            glRotatef(self.rotation.get("yaw", 0), 0, 1, 0)
            glRotatef(self.rotation.get("pitch", 0), 1, 0, 0)
            glRotatef(self.rotation.get("roll", 0), 0, 0, 1)

            # Apply scaling
            glScalef(self.scale, self.scale, self.scale)

            # Set color
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            self.model.render()
            glPopMatrix()
        except Exception as e:
            logging.error(f"Exception during plane instance rendering: {e}")


class Arrow3D:
    """
    Represents a 3D arrow used to indicate velocity vectors.
    Implemented as a line and a cone.
    """
    def __init__(self, start_pos, direction, length=50, color=QColor("red")):
        self.start_pos = start_pos  # [x, y, z]
        self.direction = direction  # [dx, dy, dz], normalized
        self.length = length
        self.color = color

    def render(self):
        try:
            end_pos = [
                self.start_pos[i] + self.direction[i] * self.length
                for i in range(3)
            ]
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex3f(*self.start_pos)
            glVertex3f(*end_pos)
            glEnd()

            # Draw the arrowhead (cone)
            glPushMatrix()
            glTranslatef(*end_pos)
            # Calculate rotation angle
            norm = math.sqrt(sum([c**2 for c in self.direction]))
            if norm != 0:
                angle = math.degrees(math.acos(self.direction[2] / norm)) if norm != 0 else 0
                axis = [-self.direction[1], self.direction[0], 0]
                if any(axis):
                    glRotatef(angle, *axis)
            glColor3f(self.color.redF(), self.color.greenF(), self.color.blueF())
            quadric = gluNewQuadric()
            gluCylinder(quadric, 40, 0.0, 40, 10, 10)
            gluDeleteQuadric(quadric)
            glPopMatrix()
        except Exception as e:
            logging.error(f"Exception during arrow rendering: {e}")


# ================================
# OpenGL Widget
# ================================
class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize dictionary to store plane instances, key is plane ID, value is Plane3DInstance
        self.planes_dict = {}
        # Initialize list to store velocity arrows, list of Arrow3D objects
        self.arrows = []
        # Last mouse position for rotating the view
        self.last_pos = None
        # Initial rotation angles for X and Y axes
        self.rotation_x = 0
        self.rotation_y = 0
        # Initial camera distance and height
        self.camera_distance = 10000.0
        self.camera_height = 10000

        self.camera_target = [0.0, 0.0, 0.0]  # Center point of the scene
        # Model cache dictionary to store loaded models and avoid reloading
        self.plane_model_cache = {}
        # Grid size range, default range is -20000 to 20000
        self.grid_size = 20000
        # Grid line spacing in pixels
        self.grid_spacing = 800

        # 定义敌方战机类型颜色映射
        self.enemy_color_map = {
            0: QColor("#22D3EE"),    # Cyan - high visibility enemy marker
            1: QColor("#60A5FA"),    # Electric blue
            2: QColor("#A78BFA"),    # Violet accent
        }

        # 新增：控制是否显示速度标签
        self.show_speed_labels = True

    def load_plane_model(self, model_path):
        # If model exists in cache, return the cached model
        if model_path in self.plane_model_cache:
            return self.plane_model_cache[model_path]
        # Otherwise, load a new model and store it in cache
        model = Plane3D(model_path, QColor("blue"))
        self.plane_model_cache[model_path] = model
        return model

    def initializeGL(self):
        try:
            # Premium dark-airspace background and smoother overlays
            glClearColor(0.015, 0.027, 0.055, 1.0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Enable depth testing for correct rendering order
            glEnable(GL_DEPTH_TEST)
            # Enable color material
            glEnable(GL_COLOR_MATERIAL)
            # Enable lighting
            glEnable(GL_LIGHTING)
            # Enable the first light source
            glEnable(GL_LIGHT0)
            # Enable normalization of normals
            glEnable(GL_NORMALIZE)
            # Set shading model to smooth
            glShadeModel(GL_SMOOTH)

            # Set light position and properties
            glLightfv(GL_LIGHT0, GL_POSITION, [0, 1000, 1000, 1])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.18, 0.22, 0.30, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.72, 0.82, 1.00, 1.0])

            # Enable texture mapping
            glEnable(GL_TEXTURE_2D)

            # Log that initialization is complete
            logging.info("OpenGL initialized successfully.")
        except Exception as e:
            # Catch and log any exceptions during initialization
            logging.error(f"OpenGL initialization failed: {e}")

    def resizeGL(self, w, h):
        # Prevent division by zero
        if h == 0:
            h = 1
        # Set the viewport to the window dimensions
        glViewport(0, 0, w, h)
        # Switch to projection matrix mode
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Use perspective projection with a 45-degree field of view
        gluPerspective(45.0, w / h, 1.0, 100000.0)
        # Switch back to modelview matrix mode
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def compute_center_point(self):
        # If there are no planes, return the origin as the center point
        if not self.planes_dict:
            return [0, 0, 0]
        # Calculate the average position of all planes as the scene's center point
        x = sum(plane.position[0] for plane in self.planes_dict.values()) / len(self.planes_dict)
        y = sum(plane.position[1] for plane in self.planes_dict.values()) / len(self.planes_dict)
        z = sum(plane.position[2] for plane in self.planes_dict.values()) / len(self.planes_dict)
        return [x, y, z]

    def paintGL(self):
        try:
            # Clear color and depth buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Calculate the center point of the scene
            center = self.compute_center_point()

            # Set the camera position
            gluLookAt(
                self.camera_target[0], self.camera_target[1] + self.camera_height,
                self.camera_target[2] + self.camera_distance,
                self.camera_target[0], self.camera_target[1], self.camera_target[2],
                0, 1, 0
            )

            # Apply scene rotation angles
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)

            # Draw coordinate axes with arrows
            self.draw_axes_with_arrows()

            # Draw ground grid
            self.draw_grid()

            # Render all plane instances
            for plane in self.planes_dict.values():
                plane.render()

            # # Render all velocity arrows
            # for arrow in self.arrows:
            #     arrow.render()

            # Finish OpenGL rendering
            glFlush()

            # Use QPainter to draw text labels
            painter = QPainter(self)
            painter.setPen(QColor("#DDE7F5"))  # Soft tactical white text
            painter.setFont(QFont('Segoe UI', 10, QFont.Weight.DemiBold))

            """
            # Draw allied planes' command information
            for plane_id, plane_instance in self.planes_dict.items():
                if not plane_instance.is_enemy:  # Only process allied planes
                    plane_pos = plane_instance.position
                    screen_pos = self.world_to_screen([plane_pos[0], plane_pos[1] + 500, plane_pos[2]])  # Raise display position
                    if screen_pos:
                        x, y = screen_pos
                        command = plane_instance.command  # Get command information
                        if command[0] == 0:
                            action_type = "Attack"
                        elif command[0] == 1:
                            action_type = "Evasion"
                        else:
                            action_type = "No Command"

                        if command[1] == -1:
                            target_info = "No Target"
                        else:
                            target_info = f"Group {command[1]}"

                        # Combine command information
                        command_text = f"{action_type} {target_info}"

                        # Draw command information
                        painter.drawText(int(x), int(y), command_text)
            """

            # Iterate through enemy planes and draw group labels and speed
            for plane_id, plane_instance in self.planes_dict.items():
                if plane_instance.is_enemy:  # Only process enemy planes
                    group = plane_instance.group if plane_instance.group is not None else "Unknown"
                    screen_pos = self.world_to_screen(plane_instance.position)
                    if screen_pos:
                        x, y = screen_pos
                        label = f"Group {group}"  # Display group label

                        # **关键修改点：将 'label' 传递给 get_group_index_from_group_label 而不是 'group'**
                        group_index = self.get_group_index_from_group_label(label)
                        color = self.enemy_color_map.get(group_index, QColor("white"))

                        # Set painter color for group label
                        painter.setPen(color)

                        # Draw group label
                        # painter.drawText(int(x), int(y), label)

                        # **新增功能：根据是否显示速度标签，绘制速度信息**
                        if self.show_speed_labels:
                            speed = plane_instance.speed
                            speed_text = f"Speed: {speed:.2f}"
                            speed_color = QColor("#FBBF24")  # Amber tactical speed label
                            painter.setPen(speed_color)
                            painter.drawText(int(x), int(y) + 15, speed_text)  # 速度标签位于组标签下方

            # Draw axis labels
            axis_labels = {"+X": [1000 + 100, 0, 0],
                           "+Y": [0, 1000 + 100, 0],
                           "+Z": [0, 0, 1000 + 100]}
            for label, pos in axis_labels.items():
                screen_pos = self.world_to_screen(pos)
                if screen_pos:
                    x, y = screen_pos
                    painter.drawText(int(x), int(y), label)

            painter.end()

        except Exception as e:
            # Catch and log any exceptions during rendering
            logging.error(f"Exception during rendering: {e}")

    def get_group_index_from_group_label(self, group_label):
        """
        Helper method to extract group index from group label string.
        Assumes group_label is in the format "Group X" where X is the index.
        """
        try:
            # 确保 group_label 是字符串
            if isinstance(group_label, int):
                # 如果 group_label 是整数，直接返回
                return group_label
            return int(group_label.split(" ")[1])
        except (IndexError, ValueError, AttributeError):
            return -1  # Return invalid index if parsing fails

    def world_to_screen(self, position):
        """
        Convert world coordinates to screen coordinates.
        """
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            win_x, win_y, win_z = gluProject(position[0], position[1], position[2], modelview, projection, viewport)
            return win_x, viewport[3] - win_y  # Flip Y-axis
        except Exception as e:
            logging.error(f"Failed to convert world coordinates to screen coordinates: {e}")
            return None

    def draw_grid(self):
        """
        Draw ground grid located at y=0 plane.
        """
        # Set line width
        glLineWidth(1.0)
        try:
            glColor4f(0.20, 0.34, 0.52, 0.35)  # Subtle blue-gray grid
            glBegin(GL_LINES)  # Start drawing lines

            # Draw lines parallel to Z-axis
            for x in range(-self.grid_size, self.grid_size + self.grid_spacing, self.grid_spacing):
                glVertex3f(x, 0, -self.grid_size)  # Start point
                glVertex3f(x, 0, self.grid_size)  # End point

            # Draw lines parallel to X-axis
            for z in range(-self.grid_size, self.grid_size + self.grid_spacing, self.grid_spacing):
                glVertex3f(-self.grid_size, 0, z)  # Start point
                glVertex3f(self.grid_size, 0, z)  # End point

            glEnd()  # End drawing
        except Exception as e:
            logging.error(f"Exception during grid drawing: {e}")

    def draw_axes_with_arrows(self, axis_length=800.0, arrow_length=150.0, arrow_radius=40.0):
        """
        Draw 3D coordinate axes with arrows.

        :param axis_length: Length of the axes
        :param arrow_length: Length of the arrowheads
        :param arrow_radius: Radius of the arrowheads
        """
        # Set line width
        glLineWidth(3.0)

        # Start drawing lines
        glBegin(GL_LINES)

        # X-axis - Red
        glColor3f(0.96, 0.27, 0.27)  # X-axis tactical red
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(axis_length, 0.0, 0.0)  # X-axis end

        # Y-axis - Green
        glColor3f(0.20, 0.83, 0.60)  # Y-axis emerald
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(0.0, axis_length, 0.0)  # Y-axis end

        # Z-axis - Blue
        glColor3f(0.13, 0.83, 0.93)  # Z-axis cyan
        glVertex3f(0.0, 0.0, 0.0)  # Origin
        glVertex3f(0.0, 0.0, axis_length)  # Z-axis end

        glEnd()

        # Start drawing arrowheads
        quadric = gluNewQuadric()  # Create a quadric object for drawing cones

        # Draw X-axis arrowhead
        glPushMatrix()  # Save current matrix
        glTranslatef(axis_length, 0.0, 0.0)  # Move to X-axis end
        glRotatef(90, 0, 1, 0)  # Rotate to align cone along X-axis
        glColor3f(0.96, 0.27, 0.27)  # X-axis tactical red
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)  # Draw cone
        glPopMatrix()  # Restore matrix

        # Draw Y-axis arrowhead
        glPushMatrix()
        glTranslatef(0.0, axis_length, 0.0)  # Move to Y-axis end
        glRotatef(-90, 1, 0, 0)  # Rotate to align cone along Y-axis
        glColor3f(0.20, 0.83, 0.60)  # Y-axis emerald
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)
        glPopMatrix()

        # Draw Z-axis arrowhead
        glPushMatrix()
        glTranslatef(0.0, 0.0, axis_length)  # Move to Z-axis end
        glColor3f(0.13, 0.83, 0.93)  # Z-axis cyan
        gluCylinder(quadric, arrow_radius, 0.0, arrow_length, 32, 32)
        glPopMatrix()

        # Delete quadric object to free resources
        gluDeleteQuadric(quadric)

    def get_plane_color_and_arrow_color(self, health_level, base_color):
        """
        Return plane color and arrow color based on health level.
        """
        if health_level > 0:
            return base_color, base_color  # Normal color
        else:
            return QColor("#64748B"), QColor("#64748B")  # Desaturated slate indicates destroyed

    def update_scene(self, ally_positions, enemy_positions):
        # Clear existing arrows
        self.arrows = []

        # Determine total number of enemy planes
        total_enemies = len(enemy_positions)

        # Preload all models to avoid loading during the loop
        f16_model_path = "models/Meshes/FixedWing.F-16.obj"
        rafale_model_path = "models/Meshes/FixedWing.Rafale.obj"
        missu_model_path = "models/Meshes/FixedWing.MQ-9.obj"

        # Load models and cache them
        f16_model = self.load_plane_model(f16_model_path)
        rafale_model = self.load_plane_model(rafale_model_path)
        missu_model = self.load_plane_model(missu_model_path)

        for index, e in enumerate(enemy_positions):
            plane_id = e.get("id", "unknown")
            pos = [e.get("x", 0), e.get("y", 0), e.get("z", 0)]
            rotation = e.get("rotation", {"pitch": 0, "yaw": 0, "roll": 0})
            vx, vy, vz = e.get("vx", 0), e.get("vy", 0), e.get("vz", 0)
            group = e.get("group", None)  # Only enemies have group info
            health_level = e.get("health_level", 1)  # Get health level
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            # Assign model based on the enemy's index to distribute them into three groups
            if total_enemies > 0:
                group_index = int(index * 3 / total_enemies)
                if group_index >= 3:
                    group_index = 2  # Ensure the index does not exceed 2
            else:
                group_index = 0  # Default to first group if no enemies

            # if group_index == 0:
            #     model = f16_model
            # elif group_index == 1:
            #     model = rafale_model
            # else:
            #     model = missu_model
            model = f16_model

            # Assign color based on group_index
            plane_color_base = self.enemy_color_map.get(0, QColor("blue"))  # Default to blue if not found

            # Adjust color based on health_level
            plane_color, arrow_color = self.get_plane_color_and_arrow_color(health_level, plane_color_base)

            # If the plane doesn't exist, add a new instance
            if plane_id not in self.planes_dict:
                self.planes_dict[plane_id] = Plane3DInstance(
                    model, pos, rotation, plane_color, group=group, is_enemy=True
                )
            else:
                # Update existing plane's position, rotation, and color
                self.planes_dict[plane_id].update_transform(pos, rotation, plane_color, group=group, speed=speed)

            # **关键修改点：设置飞机的速度**
            if plane_id in self.planes_dict:
                self.planes_dict[plane_id].speed = speed

            # Create velocity arrow and add to the list
            if speed > 0:
                direction = [vx / speed, vy / speed, vz / speed]
                arrow = Arrow3D(pos, direction, length=min(max(speed * 5, 40), 5000), color=arrow_color)
                self.arrows.append(arrow)

        # Update allied planes' positions and arrows
        for a in ally_positions:
            plane_id = a.get("id", "unknown")
            pos = [a.get("x", 0), a.get("y", 0), a.get("z", 0)]
            rotation = a.get("rotation", {"pitch": 0, "yaw": 0, "roll": 0})
            vx, vy, vz = a.get("vx", 0), a.get("vy", 0), a.get("vz", 0)
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            command = a.get("command", [-1, -1])  # Get command info
            health_level = a.get("health_level", 1)  # Get health level

            # Set plane and arrow colors based on health level
            plane_color, arrow_color = self.get_plane_color_and_arrow_color(health_level, QColor("#F43F5E"))

            if plane_id not in self.planes_dict:
                # Assuming allied planes use F-16 model; adjust if different models are needed
                model = self.load_plane_model("models/Meshes/FixedWing.F-16.obj")
                self.planes_dict[plane_id] = Plane3DInstance(
                    model, pos, rotation, plane_color, is_enemy=False
                )
            else:
                self.planes_dict[plane_id].update_transform(pos, rotation, plane_color, command=command, speed=speed)

            # **关键修改点：设置飞机的速度**
            if plane_id in self.planes_dict:
                self.planes_dict[plane_id].speed = speed

            if speed > 0:
                direction = [vx / speed, vy / speed, vz / speed]
                arrow = Arrow3D(pos, direction, length=min(max(speed * 5, 40), 5000), color=arrow_color)
                self.arrows.append(arrow)
        # Refresh the scene
        self.update()

    def mousePressEvent(self, event):
        # Record the mouse press position
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.rotation_x = max(-90, min(90, self.rotation_x + dy * 0.2))  # Limit rotation angle
            self.rotation_y += dx * 0.2
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:  # Add panning functionality
            self.camera_target[0] -= dx * 0.1
            self.camera_target[2] -= dy * 0.1
            self.update()
        self.last_pos = event.position()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 1
        self.camera_distance = max(100.0, min(self.camera_distance, 50000.0))
        self.update()

    def set_show_speed_labels(self, show: bool):
        """
        Set whether to show speed labels.
        """
        self.show_speed_labels = show
        self.update()


# ================================
# Logging Handlers
# ================================

class CustomLogFilter(logging.Filter):
    """
    Custom log filter that only allows logs containing specified keywords to pass.
    """
    def __init__(self, keywords=None):
        super().__init__()
        self.keywords = keywords if keywords else []

    def filter(self, record):
        if not self.keywords:
            return True  # Allow all logs if no keywords are specified
        return any(keyword in record.msg for keyword in self.keywords)


class QTextEditLogger(logging.Handler):
    def __init__(self, parent=None):
        super().__init__()
        self.log_widget = parent

    def emit(self, record):
        msg = self.format(record)
        self.log_widget.append(msg)


# ================================
# Main Window
# ================================

class MainWindow(QMainWindow):
    def __init__(self, client: FrontendClient):
        super().__init__()
        self.client = client

        self.setWindowTitle("Airplane Command & Preference Visualization (3D)")
        self.resize(1920, 1080)

        self.setStyleSheet("""
            /* ============================
               Premium Aerospace Command UI
               Palette:
               - Background:     #050A14 / #07111F
               - Surface:        #0B1628 / #101B2F
               - Border:         #1F3352
               - Primary Cyan:   #22D3EE
               - Electric Blue:  #3B82F6
               - Violet Accent:  #8B5CF6
               - Amber Warning:  #FBBF24
               - Success Green:  #34D399
               ============================ */

            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Roboto', 'Arial', sans-serif;
                font-size: 11px;
                color: #DDE7F5;
                background: transparent;
            }

            QMainWindow {
                background-color: #050A14;
            }

            QWidget#Root {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #050A14,
                    stop:0.50 #07111F,
                    stop:1 #0A1020
                );
            }

            QFrame#SidePanel {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0E1A2D,
                    stop:1 #091322
                );
                border: 1px solid rgba(79, 209, 255, 46);
                border-radius: 14px;
            }



            QScrollArea#SideScroll {
                background: transparent;
                border: none;
            }

            QScrollArea#SideScroll > QWidget > QWidget {
                background: transparent;
            }

            QFrame#PanelHeader {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(34, 211, 238, 41),
                    stop:0.50 rgba(59, 130, 246, 26),
                    stop:1 rgba(139, 92, 246, 33)
                );
                border: 1px solid rgba(34, 211, 238, 51);
                border-radius: 14px;
            }

            QLabel#AppTitle {
                color: #F8FBFF;
                font-size: 18px;
                font-weight: 800;
                letter-spacing: 1px;
            }

            QLabel#AppSubtitle {
                color: #8FA6C7;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }

            QLabel#SectionHint {
                color: #7C8DAA;
                font-size: 11px;
                font-weight: 500;
            }

            QLabel#ValueLabel {
                color: #22D3EE;
                font-size: 11px;
                font-weight: 700;
                padding-top: 2px;
            }

            QLabel#StatusBadge {
                color: #0B1220;
                background-color: #34D399;
                border-radius: 10px;
                padding: 7px 12px;
                font-size: 11px;
                font-weight: 800;
            }

            QLabel#ViewportTitle {
                color: #F8FBFF;
                font-size: 17px;
                font-weight: 800;
                letter-spacing: 1px;
            }

            QLabel#ViewportSubtitle {
                color: #90A4C2;
                font-size: 11px;
                font-weight: 500;
            }

            QFrame#ViewportFrame {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0C1627,
                    stop:1 #050A14
                );
                border: 1px solid rgba(59, 130, 246, 56);
                border-radius: 12px;
            }

            QFrame#ViewportTopBar {
                background: rgba(9, 19, 34, 189);
                border: 1px solid rgba(34, 211, 238, 38);
                border-radius: 14px;
            }

            QFrame#SeparatorLine {
                min-height: 1px;
                max-height: 1px;
                background-color: rgba(148, 163, 184, 41);
                border: none;
            }

            QGroupBox {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #101D32,
                    stop:1 #0B1728
                );
                border: 1px solid rgba(125, 211, 252, 41);
                border-radius: 12px;
                margin-top: 14px;
                padding: 12px 10px 10px 10px;
                color: #EAF2FF;
                font-size: 11px;
                font-weight: 800;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 16px;
                top: 2px;
                padding: 2px 10px;
                color: #22D3EE;
                background-color: #0B1728;
                border: 1px solid rgba(34, 211, 238, 51);
                border-radius: 8px;
                letter-spacing: 0.4px;
            }

            QLabel {
                color: #DDE7F5;
                font-size: 12px;
                font-weight: 600;
            }

            QTableWidget {
                background-color: rgba(7, 17, 31, 219);
                alternate-background-color: rgba(15, 31, 52, 224);
                color: #E5EEF9;
                border: 1px solid rgba(96, 165, 250, 46);
                border-radius: 12px;
                gridline-color: rgba(148, 163, 184, 26);
                font-size: 11px;
                selection-background-color: rgba(34, 211, 238, 71);
                selection-color: #FFFFFF;
                outline: 0;
            }

            QTableWidget::item {
                padding: 4px;
                border: none;
            }

            QTableWidget::item:selected {
                background-color: rgba(34, 211, 238, 82);
                color: #FFFFFF;
            }

            QHeaderView::section {
                min-height: 24px;
                max-height: 24px;
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1C2E4A,
                    stop:1 #13233C
                );
                color: #BFEFFF;
                padding: 4px;
                border: none;
                border-right: 1px solid rgba(148, 163, 184, 36);
                border-bottom: 1px solid rgba(34, 211, 238, 46);
                font-size: 11px;
                font-weight: 800;
            }

            QPushButton {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #17253C,
                    stop:1 #0E1B2E
                );
                color: #EAF2FF;
                border: 1px solid rgba(125, 211, 252, 46);
                padding: 6px 10px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 800;
                text-align: center;
            }

            QPushButton:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1E3A5F,
                    stop:1 #123253
                );
                border: 1px solid rgba(34, 211, 238, 158);
                color: #FFFFFF;
            }

            QPushButton:pressed {
                background: #0B1220;
                border: 1px solid rgba(34, 211, 238, 217);
                padding-top: 11px;
                padding-bottom: 9px;
            }

            QPushButton#PrimaryButton {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0891B2,
                    stop:1 #2563EB
                );
                border: 1px solid rgba(125, 211, 252, 140);
                color: #FFFFFF;
            }

            QPushButton#PrimaryButton:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #06B6D4,
                    stop:1 #3B82F6
                );
            }

            QPushButton#DangerButton {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7F1D1D,
                    stop:1 #B91C1C
                );
                border: 1px solid rgba(248, 113, 113, 115);
                color: #FFF1F2;
            }

            QPushButton#DangerButton:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #991B1B,
                    stop:1 #EF4444
                );
            }

            QPushButton#GhostButton {
                background: rgba(139, 92, 246, 36);
                border: 1px solid rgba(167, 139, 250, 107);
                color: #DDD6FE;
            }

            QPushButton#GhostButton:hover {
                background: rgba(139, 92, 246, 71);
                border: 1px solid rgba(196, 181, 253, 204);
                color: #FFFFFF;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0B1220,
                    stop:0.45 #123253,
                    stop:1 #1D4ED8
                );
                border: 1px solid rgba(125, 211, 252, 51);
                border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22D3EE,
                    stop:1 #8B5CF6
                );
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #F8FAFC;
                border: 3px solid #22D3EE;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }

            QSlider::handle:horizontal:hover {
                border: 3px solid #FBBF24;
                background: #FFFFFF;
            }

            QTextEdit {
                background-color: rgba(3, 7, 18, 235);
                color: #A7F3D0;
                border: 1px solid rgba(34, 211, 238, 41);
                border-radius: 12px;
                font-size: 11px;
                font-family: 'Cascadia Mono', 'JetBrains Mono', 'Consolas', monospace;
                padding: 6px;
                selection-background-color: rgba(34, 211, 238, 64);
            }

            QScrollBar:vertical {
                background: rgba(7, 17, 31, 235);
                width: 11px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:0, y2:1,
                    stop:0 #22D3EE,
                    stop:1 #3B82F6
                );
                min-height: 24px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical:hover {
                background: #FBBF24;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
                border: none;
            }

            QScrollBar:horizontal {
                background: rgba(7, 17, 31, 235);
                height: 11px;
                margin: 2px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22D3EE,
                    stop:1 #3B82F6
                );
                min-width: 24px;
                border-radius: 5px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #FBBF24;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
        """)

        # Central Widget
        central_widget = QWidget()
        central_widget.setObjectName("Root")
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # Left Control Panel: compact scrollable side panel to avoid clipping on different resolutions.
        control_scroll = QScrollArea()
        control_scroll.setObjectName("SideScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setMinimumWidth(500)
        control_scroll.setMaximumWidth(570)

        control_panel = QFrame()
        control_panel.setObjectName("SidePanel")
        control_panel.setMinimumWidth(480)

        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(14, 14, 14, 14)
        control_layout.setSpacing(10)
        control_scroll.setWidget(control_panel)
        main_layout.addWidget(control_scroll, 0)

        header_frame = QFrame()
        header_frame.setObjectName("PanelHeader")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(2)

        title_label = QLabel("AERO COMMAND CONSOLE")
        title_label.setObjectName("AppTitle")
        subtitle_label = QLabel("Human Preference · Agent Threat · 3D Tactical Situation")
        subtitle_label.setObjectName("AppSubtitle")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        control_layout.addWidget(header_frame)

        # ================================
        # Optimized "Human Preference Settings" Area
        # ================================
        pref_group = QGroupBox("Human Preference Settings")
        pref_group.setObjectName("PreferenceGroup")
        pref_layout = QVBoxLayout()
        pref_group.setLayout(pref_layout)
        pref_layout.setContentsMargins(12, 12, 12, 10)
        pref_layout.setSpacing(8)

        self.preference_widgets = []  # Store sliders and value labels
        action_labels = ["Priority"]
        group_labels = ["Enemy 1", "Enemy 2", "Enemy 3", "Enemy 4", "Enemy 5", "Enemy 6"]

        pref_hint = QLabel("Set human preference for each enemy target")
        pref_hint.setObjectName("SectionHint")
        pref_layout.addWidget(pref_hint)

        slider_grid = QGridLayout()
        slider_grid.setContentsMargins(0, 0, 0, 0)
        slider_grid.setHorizontalSpacing(8)
        slider_grid.setVerticalSpacing(6)
        pref_layout.addLayout(slider_grid)

        enemy_header = QLabel("Target")
        enemy_header.setObjectName("SectionHint")
        slider_header = QLabel("Preference")
        slider_header.setObjectName("SectionHint")
        value_header = QLabel("Value")
        value_header.setObjectName("SectionHint")
        enemy_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_grid.addWidget(enemy_header, 0, 0)
        slider_grid.addWidget(slider_header, 0, 1)
        slider_grid.addWidget(value_header, 0, 2)
        slider_grid.setColumnStretch(0, 0)
        slider_grid.setColumnStretch(1, 1)
        slider_grid.setColumnStretch(2, 0)

        for g_id, g_label in enumerate(group_labels):
            target_label = QLabel(g_label)
            target_label.setObjectName("SectionHint")
            target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target_label.setMinimumWidth(62)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(self.client.agent_preference[0][g_id] * 100))
            slider.setToolTip(f"Set Priority preference for {g_label}")
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.setObjectName("PreferenceSlider")
            slider.setFixedHeight(24)
            slider.valueChanged.connect(self.real_time_update_preference)

            value_label = QLabel(f"{self.client.agent_preference[0][g_id]:.2f}")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setObjectName("ValueLabel")
            value_label.setMinimumWidth(42)

            row = g_id + 1
            slider_grid.addWidget(target_label, row, 0)
            slider_grid.addWidget(slider, row, 1)
            slider_grid.addWidget(value_label, row, 2)
            self.preference_widgets.append((0, g_id, slider, value_label))

        # Compact action buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 4, 0, 0)
        buttons_layout.setSpacing(8)

        reset_pref_button = QPushButton("Reset Pref")
        reset_pref_button.setObjectName("DangerButton")
        reset_pref_button.setMinimumHeight(30)
        reset_pref_button.setToolTip("Click to reset all human preferences to zero")
        reset_pref_button.clicked.connect(self.reset_preferences_to_zero)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "camera2.png")
        if os.path.exists(icon_path):
            reset_pref_button.setIcon(QIcon(icon_path))
        else:
            logging.warning(f"Icon file not found: {icon_path}")
        reset_pref_button.setIconSize(QSize(16, 16))
        buttons_layout.addWidget(reset_pref_button)

        reset_camera_button = QPushButton("Reset View")
        reset_camera_button.setObjectName("PrimaryButton")
        reset_camera_button.setMinimumHeight(30)
        reset_camera_button.setToolTip("Click to reset the camera view to default position")
        reset_camera_button.clicked.connect(self.reset_camera)
        camera_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "camera1.png")
        if os.path.exists(camera_icon_path):
            reset_camera_button.setIcon(QIcon(camera_icon_path))
        else:
            logging.warning(f"Icon file not found: {camera_icon_path}")
        reset_camera_button.setIconSize(QSize(16, 16))
        buttons_layout.addWidget(reset_camera_button)

        self.toggle_speed_button = QPushButton("Speed On")
        self.toggle_speed_button.setObjectName("GhostButton")
        self.toggle_speed_button.setMinimumHeight(30)
        self.toggle_speed_button.setToolTip("Click to toggle the display of speed information")
        self.toggle_speed_button.clicked.connect(self.toggle_speed_display)
        speed_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "speed.png")
        if os.path.exists(speed_icon_path):
            self.toggle_speed_button.setIcon(QIcon(speed_icon_path))
        else:
            logging.warning(f"Icon file not found: {speed_icon_path}")
        self.toggle_speed_button.setIconSize(QSize(16, 16))
        buttons_layout.addWidget(self.toggle_speed_button)
        pref_layout.addLayout(buttons_layout)

        # Add group box to left control layout
        control_layout.addWidget(pref_group)

        # Add a separator
        separator1 = QFrame()
        separator1.setObjectName("SeparatorLine")
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Plain)
        control_layout.addWidget(separator1)

        # ================================
        # Agent Threat Levels Area
        # ================================
        threat_group = QGroupBox("Agent Threat Levels")
        threat_group.setObjectName("ThreatGroup")
        threat_layout = QVBoxLayout()
        threat_group.setLayout(threat_layout)

        # Create table to display threat levels
        self.threat_table = QTableWidget()
        self.threat_table.setColumnCount(4)
        self.threat_table.setHorizontalHeaderLabels(["Enemy ID", "Threat Level", "Enemy ID", "Threat Level"])
        self.threat_table.verticalHeader().setVisible(False)
        self.threat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.threat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Make table read-only
        self.threat_table.setShowGrid(False)
        self.threat_table.setAlternatingRowColors(True)
        self.threat_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.threat_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.threat_table.setFixedHeight(126)
        self.threat_table.horizontalHeader().setFixedHeight(26)
        self.threat_table.verticalHeader().setDefaultSectionSize(28)

        threat_layout.addWidget(self.threat_table)

        # Add threat levels group box to left control layout
        control_layout.addWidget(threat_group)

        # Add a separator
        separator2 = QFrame()
        separator2.setObjectName("SeparatorLine")
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Plain)
        control_layout.addWidget(separator2)

        # ================================
        # Status Information Area
        # ================================
        info_group = QGroupBox("Status Information")
        info_group.setObjectName("StatusGroup")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        self.connection_label = QLabel("Connection Status: {}".format("Connected" if self.client.connected else "Disconnected"))
        self.connection_label.setObjectName("StatusBadge")
        info_layout.addWidget(self.connection_label)
        self.apply_connection_badge_style()

        # Display current human preferences
        pref_table_label = QLabel("Current Mixed Preferences:")
        pref_table_label.setObjectName("SectionHint")
        info_layout.addWidget(pref_table_label)

        self.pref_table = QTableWidget(1, 6)
        self.pref_table.setHorizontalHeaderLabels(["Enemy1", "Enemy2", "Enemy3", "Enemy4", "Enemy5", "Enemy6"])
        self.pref_table.setVerticalHeaderLabels(["Priority"])
        self.pref_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pref_table.verticalHeader().setVisible(False)
        self.pref_table.setShowGrid(False)
        self.pref_table.setAlternatingRowColors(True)
        self.pref_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pref_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pref_table.setFixedHeight(74)
        self.pref_table.horizontalHeader().setFixedHeight(26)
        self.pref_table.verticalHeader().setDefaultSectionSize(28)
        self.refresh_preference_table()
        info_layout.addWidget(self.pref_table)

        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)

        # Add a separator
        separator3 = QFrame()
        separator3.setObjectName("SeparatorLine")
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setFrameShadow(QFrame.Shadow.Plain)
        control_layout.addWidget(separator3)

        # Logging Display Area
        log_group = QGroupBox("Logs")
        log_group.setObjectName("LogGroup")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setMaximumHeight(160)
        log_layout.addWidget(self.log_text)

        control_layout.addWidget(log_group)

        # Right Visualization Area (OpenGL): framed viewport with a compact tactical header.
        viewport_frame = QFrame()
        viewport_frame.setObjectName("ViewportFrame")
        viewport_layout = QVBoxLayout(viewport_frame)
        viewport_layout.setContentsMargins(12, 12, 12, 12)
        viewport_layout.setSpacing(10)

        viewport_topbar = QFrame()
        viewport_topbar.setObjectName("ViewportTopBar")
        viewport_topbar_layout = QHBoxLayout(viewport_topbar)
        viewport_topbar_layout.setContentsMargins(14, 8, 14, 8)
        viewport_topbar_layout.setSpacing(12)

        viewport_title_container = QVBoxLayout()
        viewport_title_container.setSpacing(2)
        viewport_title = QLabel("3D TACTICAL AIRSPACE")
        viewport_title.setObjectName("ViewportTitle")
        viewport_subtitle = QLabel("Left-drag: rotate · Right-drag: pan · Wheel: zoom")
        viewport_subtitle.setObjectName("ViewportSubtitle")
        viewport_title_container.addWidget(viewport_title)
        viewport_title_container.addWidget(viewport_subtitle)

        self.viewport_status_label = QLabel("LIVE RENDER · 60 FPS TARGET")
        self.viewport_status_label.setObjectName("StatusBadge")
        viewport_topbar_layout.addLayout(viewport_title_container)
        viewport_topbar_layout.addStretch(1)
        viewport_topbar_layout.addWidget(self.viewport_status_label)

        self.opengl_widget = OpenGLWidget()
        self.opengl_widget.setMinimumSize(780, 560)

        viewport_layout.addWidget(viewport_topbar)
        viewport_layout.addWidget(self.opengl_widget, 1)
        main_layout.addWidget(viewport_frame, 3)

        # Create and start data listener thread (blocking)
        self.listener_thread = DataListenerThread(self.client)
        self.listener_thread.data_received.connect(self.on_data_received)
        self.listener_thread.start()

        # Periodically update connection status (every second)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_connection_status)
        self.status_timer.start(1000)

        # Set up logging handler to output logs to the log text box
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_filter = CustomLogFilter(keywords=["Connection Status", "Camera Reset", "Sent Preferences", "Received update_positions"])
        log_handler.addFilter(self.log_filter)

        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.DEBUG)  # Set log level as needed

        # Limit the number of log lines (manual approach)
        self.max_log_lines = 100  # Maximum number of log lines in the log box

        # Start rendering timer (approximately 60FPS)
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.opengl_widget.update)
        self.render_timer.start(16)  # 16ms ≈ 60FPS

    def append_log(self, message):
        """
        Add a message to the log box while limiting the number of log lines.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log_text.append(f"[{timestamp}] {message}")

        # Check if the number of lines exceeds the maximum
        if self.log_text.document().blockCount() > self.max_log_lines:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)  # Move to the start of the text
            cursor.select(cursor.SelectionType.BlockUnderCursor)  # Select the first line
            cursor.removeSelectedText()  # Remove the selected line
            cursor.deleteChar()  # Remove the paragraph separator

    def set_log_keywords(self, keywords):
        """
        Dynamically set log filtering keywords.
        """
        self.log_filter.keywords = keywords

    def real_time_update_preference(self):
        # Real-time update of preferences
        new_pref = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        for a_id, g_id, slider, value_label in self.preference_widgets:
            val = slider.value() / 100.0
            new_pref[a_id][g_id] = val
            value_label.setText(f"{val:.2f}")
        self.client.send_preference(new_pref)
        self.client.human_preference = new_pref
        # self.refresh_preference_table()
        self.append_log(f"Sent new preferences: {new_pref}")

    def update_connection_status(self):
        self.connection_label.setText("Connection Status: {}".format("Connected" if self.client.connected else "Disconnected"))
        self.apply_connection_badge_style()

    def apply_connection_badge_style(self):
        """
        Apply a compact status-badge style according to the current backend connection state.
        """
        if self.client.connected:
            self.connection_label.setText("Connection Status: Connected")
            self.connection_label.setStyleSheet("""
                QLabel#StatusBadge {
                    color: #04130D;
                    background-color: #34D399;
                    border: 1px solid rgba(167, 243, 208, 204);
                    border-radius: 10px;
                    padding: 7px 12px;
                    font-size: 11px;
                    font-weight: 800;
                }
            """)
        else:
            self.connection_label.setText("Connection Status: Disconnected")
            self.connection_label.setStyleSheet("""
                QLabel#StatusBadge {
                    color: #FFF7ED;
                    background-color: #B91C1C;
                    border: 1px solid rgba(252, 165, 165, 204);
                    border-radius: 10px;
                    padding: 7px 12px;
                    font-size: 11px;
                    font-weight: 800;
                }
            """)

    def on_data_received(self, info):
        if info.get("type") == "update_positions":
            # 检查是否需要重置偏好
            if info.get("reset", False):
                self.reset_human_preferences()
                self.append_log("Received reset signal. Human preferences have been reset.")

            self.client.update_positions(info)
            self.refresh_preference_table()
            self.refresh_threat_table()  # 刷新威胁值表格
            # Update OpenGL visualization
            self.opengl_widget.update_scene(self.client.ally_positions, self.client.enemy_positions)
            # Update log
            num_allies = len(self.client.ally_positions)
            num_enemies = len(self.client.enemy_positions)
            if hasattr(self, "viewport_status_label"):
                self.viewport_status_label.setText(f"LIVE · ALLIES {num_allies} · ENEMIES {num_enemies}")
            self.append_log(f"Received position update: {num_allies} allies, {num_enemies} enemies.")

    def refresh_preference_table(self):
        pref = self.client.mixed_priority
        for a_id in range(1):
            for g_id in range(6):
                value = QTableWidgetItem(f"{pref[a_id][g_id]:.1f}")
                value.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pref_table.setItem(a_id, g_id, value)

    def refresh_threat_table(self):
        """
        Refresh the threat levels table based on enemy_groups data with gradient colors.
        """
        agent_preference = self.client.agent_preference
        self.threat_table.setRowCount(3)  # 设置行数
        for i in range(6):

            # 设置Group ID单元格
            item_id = QTableWidgetItem(f"Enemy{i+1}")
            item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_id.setForeground(QBrush(QColor("#BFEFFF")))
            if i < 3:
                self.threat_table.setItem(i, 0, item_id)
            else:
                self.threat_table.setItem(i-3, 2, item_id)

            # 设置Threat Level单元格
            item_threat = QTableWidgetItem(f"{agent_preference[0][i]:.4f}")
            item_threat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # 根据威胁值设置背景颜色（深色战术绿/琥珀/红）
            color = self.get_threat_color(agent_preference[0][i])
            item_threat.setBackground(QBrush(QColor(*color)))
            item_threat.setForeground(QBrush(QColor("#F8FAFC")))
            if i < 3:
                self.threat_table.setItem(i, 1, item_threat)
            else:
                self.threat_table.setItem(i-3, 3, item_threat)

    def get_threat_color(self, threat_level):
        """
        根据威胁值返回更克制的深色战术渐变：
        low -> emerald, medium -> amber, high -> rose red.
        """
        threat_level = max(0.0, min(1.0, float(threat_level)))
        if threat_level <= 0.3:
            ratio = threat_level / 0.3
            red = int(5 + 24 * ratio)
            green = int(95 + 75 * ratio)
            blue = int(80 - 10 * ratio)
        elif threat_level <= 0.7:
            ratio = (threat_level - 0.3) / 0.4
            red = int(29 + 150 * ratio)
            green = int(170 - 55 * ratio)
            blue = int(70 - 45 * ratio)
        else:
            ratio = (threat_level - 0.7) / 0.3
            red = int(179 + 55 * ratio)
            green = int(115 - 70 * ratio)
            blue = int(25 + 40 * ratio)
        return (red, green, blue)

    def reset_camera(self):
        self.opengl_widget.rotation_x = 0
        self.opengl_widget.rotation_y = 0
        self.opengl_widget.camera_distance = 10000.0
        self.opengl_widget.camera_height = 10000.0
        self.opengl_widget.camera_target = [0.0, 0.0, 0.0]
        self.opengl_widget.update()
        self.append_log("Camera has been reset to the default tactical view.")

    def reset_human_preferences(self):
        """
        Reset human preferences to default values when a reset signal is received from the backend.
        """
        default_preferences = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.client.human_preference = default_preferences
        for a_id, g_id, slider, value_label in self.preference_widgets:
            slider.blockSignals(True)  # Prevent triggering valueChanged signal
            slider.setValue(int(default_preferences[a_id][g_id] * 100))
            slider.blockSignals(False)
            value_label.setText(f"{default_preferences[a_id][g_id]:.2f}")
        self.refresh_preference_table()
        self.append_log(f"Human preferences have been reset to {default_preferences}")

    def reset_preferences_to_zero(self):
        """
        Reset all human preferences to 0 and update the UI.
        """
        default_preferences = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.client.human_preference = default_preferences
        for a_id, g_id, slider, value_label in self.preference_widgets:
            slider.blockSignals(True)  # Prevent triggering valueChanged signal
            slider.setValue(int(default_preferences[a_id][g_id] * 100))
            slider.blockSignals(False)
            value_label.setText(f"{default_preferences[a_id][g_id]:.2f}")
        self.refresh_preference_table()
        self.client.send_preference(default_preferences)
        self.append_log(f"All human preferences have been reset to {default_preferences}")

    def toggle_speed_display(self):
        """
        Toggle the display of speed information on enemy planes.
        """
        if self.opengl_widget.show_speed_labels:
            self.opengl_widget.set_show_speed_labels(False)
            self.toggle_speed_button.setText("Speed Off")
            self.append_log("Speed display has been hidden.")
        else:
            self.opengl_widget.set_show_speed_labels(True)
            self.toggle_speed_button.setText("Speed On")
            self.append_log("Speed display has been shown.")

    def closeEvent(self, event):
        # Stop listener thread and close connection when window closes
        if self.listener_thread.isRunning():
            self.listener_thread.stop()
            self.listener_thread.wait()
        self.client.close()
        event.accept()


# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        client = FrontendClient('127.0.0.1', 9999)
        window = MainWindow(client)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Unhandled exception during program execution: {e}")
