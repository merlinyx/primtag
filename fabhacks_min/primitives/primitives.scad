$fn=30;

module rod(radius, length) {
    linear_extrude(height=length, convexity=10, center=true) circle(radius);
}

//rod(1, 10);

module hook(arc_radius, arc_angle, thickness) {
    rotate([0,90,180]) rotate_extrude(angle=arc_angle, convexity=10) translate([arc_radius, 0]) circle(thickness);
}

//hook(15, 270, 5);

//will not use ring because it won't render for unknown reasons
module ring(arc_radius, thickness) {
    hook(arc_radius, 360, thickness);
}

//ring(35, 5);

module tube(inner_radius, thickness, length) {
    difference() {
        rod(inner_radius + thickness, length);
        rod(inner_radius, length);
    }
}

//tube(5, 1, 20);

module edge(width, length, height, center = true) {
    rotate([0, 0, 90]) linear_extrude(height=height, center=center) square([width, length], center=true);
}

//edge(10, 20, 5);

module surface(width, length, height=0.1) {
    edge(width, length, height);
}

//surface(10, 20);

module clip(width, height, thickness, dist, open_gap) {
    angle = atan((open_gap - dist) / height / 2.0);
    w = height / cos(angle);
    l = width;
    union() {
        translate([(dist + open_gap) / 4.0, 0, 0])
        rotate([-angle-90, 0, 90]) edge(w, l, thickness);//, center = false);
        translate([-(dist + open_gap) / 4.0, 0, 0])
        rotate([angle+90, 0, 90]) edge(w, l, thickness);//, center = false);
    }
}

//clip(20, 15, 1, 15, 10);

module hemisphere(radius) {
    difference() {
        sphere(radius);
        edge(radius*2, radius*2, radius*2, center=false);
    }
}

//hemisphere(20);

module point() {
    cube([1, 1, 1], center=true);
}

//point();
