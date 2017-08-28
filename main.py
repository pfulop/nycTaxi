import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime

train = pd.read_csv('./inputs/train.csv')


def stripTime():
    t = train['pickup_datetime'].apply(
        lambda d: pd.Series([datetime.strptime(d, "%Y-%m-%d %H:%M:%S").strftime("%H"),
                   datetime.strptime(d, "%Y-%m-%d %H:%M:%S").strftime("%d%m")]))
    print(t)


stripTime()

fig, (ax_pup, ax_doff) = plt.subplots(1, 2)
plt.tight_layout()

ax_pup.set_xlim(-74.1, -73.7)
ax_pup.set_ylim(40.60, 40.90)
ax_pup.set_title("Pickup")

ax_doff.set_xlim(-74.1, -73.7)
ax_doff.set_ylim(40.60, 40.90)
ax_doff.set_title("Dropoff")

markers_pup, = ax_pup.plot([], [], '.', markersize=0.05)
markers_doff, = ax_doff.plot([], [], 'r.', markersize=0.05)

time_text = ax_pup.text(0.05, 0.05, '', transform=ax_pup.transAxes)


def getTimePeriod(i):
    return train[train['pickup_datetime'] == str(i).zfill(2)]


def init():
    markers_doff.set_data([], [])
    markers_pup.set_data([], [])
    return markers_doff, markers_pup,


def animate(i):
    anim_data = getTimePeriod(i)
    markers_doff.set_data(anim_data.dropoff_longitude, anim_data.dropoff_latitude)
    markers_pup.set_data(anim_data.pickup_longitude, anim_data.pickup_latitude)
    time_text.set_text('%02d H' % i)
    return markers_doff, markers_pup, time_text,


print(train.groupby(['pickup_datetime'])['trip_duration'].mean())

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=24, interval=40, blit=True)

plt.show()
