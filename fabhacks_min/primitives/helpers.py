def clamp(t, tmin = 0., tmax = 1.):
    try:
        assert t >= tmin and t <= tmax
    except AssertionError:
        if t < tmin: t = tmin
        if t > tmax: t = tmax
    return t

def clamp_angle(a, amin, amax):
    a = a % 360.
    try:
        assert a >= amin and a <= amax
    except AssertionError:
        if a < amin: a = amin
        if a > amax: a = amax
    return a
