// These macros redefine the CUDA blocks and grids to be row-major,
// instead of column major.

#define _tx threadIdx.z
#define _ty (signed int)(threadIdx.y - {{ padding[0] }})
#define _tz (signed int)(threadIdx.x - {{ padding[2] }})

#define _bx blockIdx.z
#define _by blockIdx.y
#define _bz blockIdx.x

#define _txx blockDim.z
#define _tyy (blockDim.y - {{ padding[0] + padding[1] }})
#define _tzz (blockDim.x - {{ padding[2] + padding[3] }})

#define _bxx gridDim.z
#define _byy gridDim.y
#define _bzz gridDim.x

// Use the complex-value definition and operators included with pycuda.
// This allows us to work with pycuda's GPUArray class.
#include <pycuda-complex.hpp>

// Defines row major access to a 3D array.
// dx, dy, dz are shifts from the present location of the field.
// Note that there is an offset in the x-index
#define _MY_OFFSET(dx,dy,dz) ((_X - {{ x_range[0] }} + dx) * {{ dims[1] }} * {{ dims[2] }} + \
                             (_Y + dy) * {{ dims[2] }} + (_Z + dz))

// Macros to access fields using the field(i,j,k) format,
// where sx, sy, and sz are RELATIVE offsets in the x, y, and z directions
// respectively.
{%- for p in params if p.gce_type == 'grid' %}
#define {{ p.name }}(dx,dy,dz) {{ p.name }}[_MY_OFFSET(dx,dy,dz)]
{%- endfor %} 

// Constants. We have to have a crude work-around to avoid problems with 
// trying to use pycuda::complex types in constant memory.
{# Commented out for now.
{%- for p in params if p.gce_type == 'const' %}
__constant__ {{ p.alt_type }}  _{{ p.name }}_temp[{{ p.num_elems }}];
{%- endfor %} 
{%- for p in params if p.gce_type == 'const' %}
{%- if p.cuda_type in ('pycuda::complex<double>', 'pycuda::complex<float>') %}
#define {{ p.name }}(i) {{ p.cuda_type }}(_{{ p.name }}_temp[i].x, _{{ p.name }}_temp[i].y)
{%- else %}
#define {{ p.name }}(i) _{{ p.name }}_temp[i]
{%- endif %}
{%- endfor %} 
#}
{%- for p in params if p.gce_type == 'const' %}
#define {{ p.name }}(i) {{ p.name }}[i]
{%- endfor %} 

// Dynamic allocation of shared memory
extern __shared__ pycuda::complex<double> _gce_smem[];

__global__ void _gce_kernel(const int _x_start, const int _x_end, 
        {#- Add the fields as input parameters to the function. -#}
                           {#{%- for p in params if p.gce_type != 'const' -%}#}
                           {%- for p in params -%} 
                           {% if p.gce_type ==  'const' -%}
                           {{ p.cuda_type }}* {{ p.name }}
                           {% elif p.gce_type ==  'number' -%}
                           {{ p.cuda_type }} {{ p.name }}
                           {% elif p.gce_type == 'out' -%}
                           {{ p.cuda_type }} *_{{ p.name }}_out
                           {% else -%} 
                           {{ p.cuda_type }}* {{ p.name }}
                           {% endif -%}
                           {%- if not loop.last -%}, 
                           {%- else -%}) {% endif %} {% endfor %} 
{
    // Global index variables which determine where you are in the space,
    // and subsequently which grid point you will access.
    int _X = _tx + _txx * _bx + _x_start;
    int _Y = _ty + _tyy * _by;
    int _Z = _tz + _tzz * _bz;

    // Threads that are responsible for a grid point.
    const bool _in_global = (((_Y >= 0) && (_Y < {{ dims[1] }})) && \
                            ((_Z >= 0) && (_Z < {{ dims[2] }})));

    // Threads that are not part of the thread block padding.
    const bool _in_local =  (_ty >= 0) && (_ty < _tyy) && \
                            (_tz >= 0) && (_tz < _tzz);

    // Initialize the local variables for the Outs.
    {%- for p in params if p.gce_type == 'out' %}
    // {{ p.cuda_type }} {{ p.name }} = {{ p.cuda_type }}(0);
    {{ p.cuda_type }} {{ p.name }} = _{{ p.name }}_out[_Y * {{ dims[2] }} + _Z];
    {%- endfor %} 

    // User-defined "pre-loop" code.
    {{ preloop_code }}

    for (; _X < _x_end ; _X += _txx) {
        // User-defined "loop" code.
        {{ loop_code }}
    }
   
    // Save outs of non-padding threads to global memory.
    if (_in_local && _in_global) {
        {%- for p in params if p.gce_type == 'out' %}
        _{{ p.name }}_out[_Y * {{ dims[2] }} + _Z] = {{ p.name }};
        {%- endfor %} 
    }

    return;
}



