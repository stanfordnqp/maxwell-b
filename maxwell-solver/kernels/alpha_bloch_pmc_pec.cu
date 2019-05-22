// Mark the threads that need to load from global memory.
const bool adj_dims =  (((_X >= -1) && (_X <= {{ dims[0] }})) && \
                        ((_Y >= -1) && (_Y <= {{ dims[1] }})) && \
                        ((_Z >= -1) && (_Z <= {{ dims[2] }})));

// Set relevant field pointers to create wrap-around periodic grid.
{{ type }} bloch_phaseYZ_x = 1.0;
{{ type }} bloch_phaseYZ_y = 1.0;
{{ type }} bloch_phaseYZ_z = 1.0;

{{ type }} pc_yz_factor[3] = { 1.0, 1.0, 1.0 };
int pc_iy_Ex = 0;
int pc_iz_Ex = 0;
int pc_iy_Ey = 0;
int pc_iz_Ey = 0;
int pc_iy_Ez = 0;
int pc_iz_Ez = 0;

if (_Y == -1) {
    _Y = {{ dims[1]-1 }};
	bloch_phaseYZ_x *= bloch_y(0);
	bloch_phaseYZ_y *= bloch_y(1);
	bloch_phaseYZ_z *= bloch_y(2);
	if ( pemc(2) == 1 ) {
        //PEC (anti-symmetric)
        _Y = 0;
		pc_yz_factor[0] = -1.0;
		pc_yz_factor[2] = -1.0;
		pc_iy_Ex = 1;
		pc_iy_Ey = 0;
		pc_iy_Ez = 1;
	} else if ( pemc(2) == 2 ) {
        //PMC (symmetric)
        _Y = 0;
		pc_yz_factor[1] = -1.0;
		pc_iy_Ex = 1;
		pc_iy_Ey = 0;
		pc_iy_Ez = 1;
	}
}
if (_Y == {{ dims[1]-1 }}) {
	if ( pemc(3) == 1 ) {
		pc_iy_Ey = -1;
	}
	if ( pemc(3) == 2 ) {
		pc_iy_Ey = -1;
		pc_yz_factor[1] = -1.0;
	}
}
if (_Y == {{ dims[1] }}) {
    _Y = 0;
	bloch_phaseYZ_x *= conj(bloch_y(0));
	bloch_phaseYZ_y *= conj(bloch_y(1));
	bloch_phaseYZ_z *= conj(bloch_y(2));
	if ( pemc(3) == 1 ) {
        //PEC
        _Y = {{ dims[1]-1 }};
		pc_yz_factor[0] = -1.0;
		pc_yz_factor[2] = -1.0;
		pc_iy_Ex = -1;
		pc_iy_Ey = 0;
		pc_iy_Ez = -1;
	}
    if ( pemc(3) == 2 ) {
        //PMC
        _Y = {{ dims[1]-1 }};
		pc_yz_factor[1] = -1.0;
		pc_iy_Ex = -1;
		pc_iy_Ey = 0;
		pc_iy_Ez = -1;
	}
}
if (_Z == -1) {
    _Z = {{ dims[2]-1 }};
	bloch_phaseYZ_x *= bloch_z(0);
	bloch_phaseYZ_y *= bloch_z(1);
	bloch_phaseYZ_z *= bloch_z(2);
	if ( pemc(4) == 1 ) {
        //PEC (anti-symmetric)
        _Z = 0;
		pc_yz_factor[0] = -1.0;
		pc_yz_factor[1] = -1.0;
		pc_iz_Ex = 1;
		pc_iz_Ey = 1;
		pc_iz_Ez = 0;
	} else if ( pemc(4) == 2 ) {
        //PMC (symmetric)
        _Z = 0;
		pc_yz_factor[2] = -1.0;
		pc_iz_Ex = 1;
		pc_iz_Ey = 1;
		pc_iz_Ez = 0;
	}
}
if (_Z == {{ dims[2]-1 }}) {
	if ( pemc(5) == 1 ) {
		pc_iz_Ez = -1;
	}
	if ( pemc(5) == 2 ) {
		pc_iz_Ez = -1;
		pc_yz_factor[2] = -1.0;
	}
}
if (_Z == {{ dims[2] }}) {
    _Z = 0;
	bloch_phaseYZ_x *= conj(bloch_z(0));
	bloch_phaseYZ_y *= conj(bloch_z(1));
	bloch_phaseYZ_z *= conj(bloch_z(2));
	if ( pemc(5) == 1 ) {
        //PEC
        _Z = {{ dims[2]-1 }};
		pc_yz_factor[0] = -1.0;
		pc_yz_factor[1] = -1.0;
		pc_iz_Ex = -1;
		pc_iz_Ey = -1;
		pc_iz_Ez = 0;
	} else if ( pemc(5) == 2 ) {
        //PMC
        _Z = {{ dims[2]-1 }};
		pc_yz_factor[2] = 0.0; //this value does not matter
		pc_iz_Ex = -1;
		pc_iz_Ey = -1;
		pc_iz_Ez = 0;
	}
}

// Some definitions for shared memory.
// Used to get unpadded thread indices.
#define s_ty (_ty + 1)
#define s_tz (_tz + 1)
#define s_tyy (_tyy + 2)
#define s_tzz (_tzz + 2)

// Helper definitions.
#define s_next_field (s_tyy * s_tzz)
#define s_to_local (s_ty * s_tzz + (s_tz))   
#define s_zp +1
#define s_zn -1
#define s_yp +s_tzz
#define s_yn -s_tzz

{{ type }} *Ex_0 = (0 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;
{{ type }} *Ey_0 = (1 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;
{{ type }} *Ez_0 = (2 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;
{{ type }} *Hx_0 = (3 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;
{{ type }} *Hy_0 = (4 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;
{{ type }} *Hz_0 = (5 * s_next_field) + (({{ type }}*) _gce_smem) + s_to_local;

// Local memory.
{{ type }} Ey_p, Ez_p, Hy_n, Hz_n;
{{ type }} vx, vy, vz;
{{ type }} px, py, pz, py_p, pz_p;

int xn, xp;
{{ type }} bloch_phaseX_x = 1;
{{ type }} bloch_phaseX_y = 1;
{{ type }} bloch_phaseX_z = 1;
{{ type }} pc_x_factor[3] = { 1.0, 1.0, 1.0 };
int pc_ix_Ex = -1;
int pc_ix_Ey = -1;
int pc_ix_Ez = -1;
if (_X == 0) { 
    bloch_phaseX_x = bloch_x(0);
    bloch_phaseX_y = bloch_x(1);
    bloch_phaseX_z = bloch_x(2);
    if ( pemc(0) == 1 ) {
        pc_x_factor[1] = -1.0;
        pc_x_factor[2] = -1.0;
        pc_ix_Ex = 0;
        pc_ix_Ey = 1;
        pc_ix_Ez = 1;
        xn = 0;
    } else if ( pemc(0) == 2 ) {
        pc_x_factor[0] = -1.0;
        pc_ix_Ex = 0;
        pc_ix_Ey = 1;
        pc_ix_Ez = 1;
        xn = 0;
    } else {
        pc_ix_Ex = -1;
        pc_ix_Ey = -1;
        pc_ix_Ez = -1;
        xn = {{ dims[0]-1 }}; // Wrap-around step in the negative direction.
    }
} else {
    xn = -1;}

// Load E-fields into shared memory.
if (adj_dims) {
    // Load in p = r + beta * p.
    Ex_0[0] = bloch_phaseX_x * bloch_phaseYZ_x * pc_x_factor[0] * pc_yz_factor[0] *
                (sqrt_sx1(_X+xn) * sqrt_sy0(_Y) * sqrt_sz0(_Z)) *
	        (  Rx(pc_ix_Ex, pc_iy_Ex, pc_iz_Ex) 
		 + beta * ( Px(pc_ix_Ex, pc_iy_Ex, pc_iz_Ex) 
			    - omega * Vx(pc_ix_Ex, pc_iy_Ex, pc_iz_Ex)));
    Ey_0[0] = bloch_phaseX_y * bloch_phaseYZ_y * pc_x_factor[1] * pc_yz_factor[1] *
                (sqrt_sx0(_X+xn) * sqrt_sy1(_Y) * sqrt_sz0(_Z)) *
	        (  Ry(pc_ix_Ey, pc_iy_Ey, pc_iz_Ey)
		 + beta * ( Py(pc_ix_Ey, pc_iy_Ey, pc_iz_Ey) 
			    - omega * Vy(pc_ix_Ey, pc_iy_Ey, pc_iz_Ey)));
    Ez_0[0] = bloch_phaseX_z * bloch_phaseYZ_z * pc_x_factor[2] * pc_yz_factor[2] *
                (sqrt_sx0(_X+xn) * sqrt_sy0(_Y) * sqrt_sz1(_Z)) *
	        (  Rz(pc_ix_Ez, pc_iy_Ez, pc_iz_Ez) 
		 + beta * ( Pz(pc_ix_Ez, pc_iy_Ez, pc_iz_Ez) 
			    - omega * Vz(pc_ix_Ez, pc_iy_Ez, pc_iz_Ez)));

    // Ey_p = Ry(0,0,0) + beta * Ey(0,0,0);
    py_p = Ry(0, pc_iy_Ey, pc_iz_Ey) 
            + beta * (Py(0, pc_iy_Ey, pc_iz_Ey) 
                        - omega * Vy(0, pc_iy_Ey, pc_iz_Ey));
    Ey_p = bloch_phaseYZ_y * pc_yz_factor[1] * (py_p) * 
            (sqrt_sx0(_X) * sqrt_sy1(_Y) * sqrt_sz0(_Z));

    // Ez_p = Rz(0,0,0) + beta * Ez(0,0,0);
    pz_p = Rz(0, pc_iy_Ez, pc_iz_Ez) 
            + beta * (Pz(0, pc_iy_Ez, pc_iz_Ez)
                        - omega * Vz(0, pc_iy_Ez, pc_iz_Ez));
    Ez_p = bloch_phaseYZ_z * pc_yz_factor[2] * (pz_p) * 
            (sqrt_sx0(_X) * sqrt_sy0(_Y) * sqrt_sz1(_Z));
}
__syncthreads();

// Calculate H-fields and store in shared_memory.
// Hy.
if ((_ty != -1) && (_ty != _tyy) && (_tz != _tzz)) {
    Hy_0[0] = my(pc_ix_Ey, pc_iy_Ey, pc_iz_Ey) * 
                (sx1(_X+xn) * (Ez_0[0] - Ez_p) - sz1(_Z) * (Ex_0[0] - Ex_0[s_zp]));
}

// Hz.
if ((_ty != _tyy) && (_tz != -1) && (_tz != _tzz)) {
    Hz_0[0] = mz(pc_ix_Ez, pc_iy_Ez, pc_iz_Ez) * 
                (sy1(_Y) * (Ex_0[0] - Ex_0[s_yp]) - sx1(_X+xn) * (Ey_0[0] - Ey_p));
}
__syncthreads();

// reset the pemc factors and ix's
pc_x_factor[0] = 1.0;
pc_x_factor[1] = 1.0;
pc_x_factor[2] = 1.0;
pc_ix_Ex = 0;
pc_ix_Ey = 1;
pc_ix_Ez = 1;
// start loop in x direction
for (; _X < _x_end ; _X += _txx) {
    // We've moved ahead in X, so transfer appropriate field values.
    Ey_0[0] = Ey_p;
    Ez_0[0] = Ez_p;
    Hy_n = Hy_0[0];
    Hz_n = Hz_0[0];

    py = py_p;
    pz = pz_p;

    // Load E-fields into shared memory.
    if (_X == {{ dims[0]-1 }}){
        if ( pemc(1) == 1 ) {
            // PEC
            pc_x_factor[1] = -1.0;
            pc_x_factor[2] = -1.0;
            pc_ix_Ex = -1;
            pc_ix_Ey = -1;
            pc_ix_Ez = -1;
            xp = 0;
        } else if ( pemc(1) == 2 ) {
            // PMC
            pc_x_factor[0] = -1.0;
            pc_ix_Ex = -1;
            pc_ix_Ey = -1;
            pc_ix_Ez = -1;
            xp = 0;
        } else {
            // bloch
            bloch_phaseX_x = conj(bloch_x(0));
            bloch_phaseX_y = conj(bloch_x(1));
            bloch_phaseX_z = conj(bloch_x(2));
            xp = {{ -(dims[0]-1) }}; // Wrap-around step in the negative direction.
        }
    } else {
        xp = +1;
    	bloch_phaseX_x = 1;
    	bloch_phaseX_y = 1;
    	bloch_phaseX_z = 1;
    }

    if (adj_dims) {
        px = Rx(pc_ix_Ex,pc_iy_Ex,pc_iz_Ex) + beta * ( 
			Px(pc_ix_Ex,pc_iy_Ex,pc_iz_Ex) - omega * Vx(pc_ix_Ex,pc_iy_Ex,pc_iz_Ex));
        Ex_0[0] = bloch_phaseYZ_x * pc_x_factor[0] * pc_yz_factor[0] *
                    (px) * (sqrt_sx1(_X) * sqrt_sy0(_Y) * sqrt_sz0(_Z));

        py_p = Ry(pc_ix_Ey,pc_iy_Ey,pc_iz_Ey) + beta * ( 
			Py(pc_ix_Ey,pc_iy_Ey,pc_iz_Ey) - omega * Vy(pc_ix_Ez,pc_iy_Ey,pc_iz_Ey));
        Ey_p = bloch_phaseX_y * bloch_phaseYZ_y * pc_x_factor[1] * pc_yz_factor[1] *
                    (py_p) * (sqrt_sx0(_X+xp) * sqrt_sy1(_Y) * sqrt_sz0(_Z));

        pz_p = Rz(pc_ix_Ez,pc_iy_Ez,pc_iz_Ez) + beta * ( 
			Pz(pc_ix_Ez,pc_iy_Ez,pc_iz_Ez) - omega * Vz(pc_ix_Ez,pc_iy_Ez,pc_iz_Ez));
        Ez_p = bloch_phaseX_z * bloch_phaseYZ_z * pc_x_factor[2] * pc_yz_factor[2] *
                    (pz_p) * (sqrt_sx0(_X+xp) * sqrt_sy0(_Y) * sqrt_sz1(_Z));
    }

    __syncthreads();

    // Calculate H-fields and store in shared_memory.
    {% if mu_equals_1 == True %}
    // Hx.
    if ((_ty != _tyy) && (_tz != _tzz)) {
        Hx_0[0] =   (sz1(_Z) * (Ey_0[0] - Ey_0[s_zp]) - 
                    sy1(_Y) * (Ez_0[0] - Ez_0[s_yp]));
    }

    // Hy.
    if ((_ty != -1) && (_ty != _tyy) && (_tz != _tzz)) {
        Hy_0[0] =   (sx1(_X) * (Ez_0[0] - Ez_p) - 
                    sz1(_Z) * (Ex_0[0] - Ex_0[s_zp]));
    }

    // Hz.
    if ((_ty != _tyy) && (_tz != -1) && (_tz != _tzz)) {
        Hz_0[0] =   (sy1(_Y) * (Ex_0[0] - Ex_0[s_yp]) - 
                    sx1(_X) * (Ey_0[0] - Ey_p));
    }
    {% else %}
    // Hx.
    if ((_ty != _tyy) && (_tz != _tzz)) {
        Hx_0[0] =   mx(0,0,0) * (sz1(_Z) * (Ey_0[0] - Ey_0[s_zp]) - 
                    sy1(_Y) * (Ez_0[0] - Ez_0[s_yp]));
    }

    // Hy.
    if ((_ty != -1) && (_ty != _tyy) && (_tz != _tzz)) {
        Hy_0[0] =   my(0,0,0) * (sx1(_X) * (Ez_0[0] - Ez_p) - 
                    sz1(_Z) * (Ex_0[0] - Ex_0[s_zp]));
    }

    // Hz.
    if ((_ty != _tyy) && (_tz != -1) && (_tz != _tzz)) {
        Hz_0[0] =   mz(0,0,0) * (sy1(_Y) * (Ex_0[0] - Ex_0[s_yp]) - 
                    sx1(_X) * (Ey_0[0] - Ey_p));
    }
    {% endif %}
    __syncthreads();

    // Write out the results.
    if (_in_global && _in_local) {
        {% if full_operator %}
        P1x(0,0,0) = px;
        P1y(0,0,0) = py;
        P1z(0,0,0) = pz;

        vx = ((1.0 / (sqrt_sx1(_X) * sqrt_sy0(_Y) * sqrt_sz0(_Z))) *
                    (sy0(_Y) * (Hz_0[0] - Hz_0[s_yn])
                    - sz0(_Z) * (Hy_0[0] - Hy_0[s_zn])
                    - ex(0,0,0) * Ex_0[0]));
        vy = ((1.0 / (sqrt_sx0(_X) * sqrt_sy1(_Y) * sqrt_sz0(_Z))) *
                    (sz0(_Z) * (Hx_0[0] - Hx_0[s_zn]) 
                    - sx0(_X) * (Hz_0[0] - Hz_n) 
                    - ey(0,0,0) * Ey_0[0]));
        vz = ((1.0 / (sqrt_sx0(_X) * sqrt_sy0(_Y) * sqrt_sz1(_Z))) *
                    (sx0(_X) * (Hy_0[0] - Hy_n) 
                    - sy0(_Y) * (Hx_0[0] - Hx_0[s_yn]) 
                    - ez(0,0,0) * Ez_0[0]));

        V1x(0,0,0) = vx;
        V1y(0,0,0) = vy;
        V1z(0,0,0) = vz;

        alpha_denom += (R_hatHx(0,0,0) * vx) + (R_hatHy(0,0,0) * vy) + (R_hatHz(0,0,0) * vz);

        {% else %}
        V1x(0,0,0) = Hx_0[0];
        V1y(0,0,0) = Hy_0[0];
        V1z(0,0,0) = Hz_0[0];

        {% endif %}
    }
    __syncthreads();
}
