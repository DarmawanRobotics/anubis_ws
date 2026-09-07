FROM rwthika/ros2:humble-desktop-full

# 
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG WORKSPACE_NAME=ros2_ws

# Set environment variables to avoid interactive prompts during package installation and configure display settings for GUI applications
ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    LIBGL_ALWAYS_SOFTWARE=1

# Install necessary packages for GUI, VNC, SSH, and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ubuntu-desktop-minimal \
        xfce4 \
        xfce4-terminal \
        tigervnc-standalone-server \
        tigervnc-common \
        x11vnc \
        xclip \
        dbus-x11 \
        sudo \
        usbutils \
        udev \
        net-tools \
        openssh-server \
        tmux \
        htop \
        git \
        python3-pip \
    && (apt-get purge -y gdm3 snapd || true) \
    && apt-get autoremove -y \
    && pip3 install --no-cache-dir tmuxp \
    && mkdir -p /var/run/sshd \
    && sed -i 's/^session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/' /etc/pam.d/sshd

# Create a non-root user with specified UID and GID, add to necessary groups, configure sudoers, and set up SSH and VNC directories
RUN (getent group $USER_GID > /dev/null 2>&1 || groupadd --gid $USER_GID $USERNAME) \
    && useradd --non-unique --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && usermod -aG sudo,dialout,plugdev $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME/.ssh \
    && chmod 700 /home/$USERNAME/.ssh \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/.ssh

# Create a workspace directory for ROS 2 and set appropriate ownership
RUN mkdir -p /home/$USERNAME/$WORKSPACE_NAME \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/$WORKSPACE_NAME

# Create ~/.ros with correct ownership up front — this dir will be a named
# volume (rosdep cache), and Docker copies the image's existing content
# (incl. ownership) into a named volume the first time it's mounted, so
# creating it here avoids the volume showing up root-owned instead of
# owned by $USERNAME.
RUN mkdir -p /home/$USERNAME/.ros \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/.ros

# Disable autostart for certain applications to prevent them from launching automatically in the desktop environment 
RUN mkdir -p /home/$USERNAME/.config/autostart \
    && printf '[Desktop Entry]\nHidden=true\n' > /home/$USERNAME/.config/autostart/tracker-miner-fs-3.desktop \
    && printf '[Desktop Entry]\nHidden=true\n' > /home/$USERNAME/.config/autostart/tracker-extract-3.desktop \
    && printf '[Desktop Entry]\nHidden=true\n' > /home/$USERNAME/.config/autostart/update-notifier.desktop \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/.config

# Copy necessary configuration files for entrypoint, VNC, SSH, and tmux, set permissions, and configure the user's environment
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY vnc-xstartup /home/$USERNAME/.vnc/xstartup.template
COPY sshd_config /etc/ssh/sshd_config
COPY tmux/tmux.conf /home/$USERNAME/.tmux.conf
COPY tmux/session.yaml /home/$USERNAME/tmux/session.yaml

# Set permissions for the entrypoint script, adjust ownership of configuration files, and update the user's bashrc with ROS 2 setup and tmux aliases
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown $USERNAME:$USER_GID /home/$USERNAME/.vnc/xstartup.template /home/$USERNAME/.tmux.conf \
    && chown -R $USERNAME:$USER_GID /home/$USERNAME/tmux \
    && sed -i "s/__USERNAME__/$USERNAME/g" /home/$USERNAME/tmux/session.yaml \
    && sed -i "s/__WORKSPACE_NAME__/$WORKSPACE_NAME/g" /home/$USERNAME/tmux/session.yaml /home/$USERNAME/.tmux.conf \
    && echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /home/$USERNAME/$WORKSPACE_NAME/install/setup.bash ] && source /home/$USERNAME/$WORKSPACE_NAME/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "alias tmux-start='tmuxp load -y ~/tmux/session.yaml'" >> /home/$USERNAME/.bashrc \
    && echo "alias tmux-kill='tmux kill-session -t $WORKSPACE_NAME'" >> /home/$USERNAME/.bashrc \
    && echo "[ \"\$PWD\" = \"/home/$USERNAME\" ] && cd /home/$USERNAME/$WORKSPACE_NAME" >> /home/$USERNAME/.bashrc


WORKDIR /home/$USERNAME/$WORKSPACE_NAME
EXPOSE 5901 2222

# sshd is the one service the entrypoint always tries to keep up (VNC can
# fail to start and the container stays "running" anyway per entrypoint.sh),
# so it's the most honest single signal for "is this container actually usable".
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD bash -c 'echo > /dev/tcp/127.0.0.1/2222' || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []