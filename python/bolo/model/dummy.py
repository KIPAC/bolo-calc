#!/usr/bin/env python
"""
Classes used to describe aspect of Models.

The base class is `Property` which describes any one property of a model,
such as the name, or some other fixed property.

The `Parameter` class describes variable model parameters.

The `Derived` class describes model properies that are derived
from other model properties.

"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy

from abc import ABC, abstractmethod

from numbers import Number
from collections import OrderedDict as odict

import numpy as np
import yaml

try:
    basestring
except NameError:
    basestring = str


def is_none(val):
    """Check for none as string"""
    return val in [None, 'none', 'None']


def asscalar(a):
    """Convert single-item lists and numpy arrays to scalars. Does
    not care about the type of the elements (i.e., will work fine on
    strings, etc.)

    https://github.com/numpy/numpy/issues/4701
    https://github.com/numpy/numpy/pull/5126
    """
    try:
        return a.item()
    except AttributeError:
        return np.asarray(a).item()


def defaults_docstring(defaults, header=None, indent=None, footer=None):
    """Return a docstring from a list of defaults.
    """
    if indent is None:
        indent = ''
    if header is None:
        header = ''
    if footer is None:
        footer = ''

    #width = 60
    #hbar = indent + width * '=' + '\n'  # horizontal bar
    hbar = '\n'

    s = hbar + (header) + hbar
    for key, value, desc in defaults:
        if isinstance(value, basestring):
            value = "'" + value + "'"
        if hasattr(value, '__call__'):
            value = "<" + value.__name__ + ">"

        s += indent +'%-12s\n' % ("%s :" % key)
        s += indent + indent + (indent + 23 * ' ').join(desc.split('\n'))
        s += ' [%s]\n\n' % str(value)
    s += hbar
    s += footer
    return s


def defaults_decorator(defaults):
    """Decorator to append default kwargs to a function.
    """
    def decorator(func):
        """Function that appends default kwargs to a function.
        """
        kwargs = dict(header='Keyword arguments\n-----------------\n',
                      indent='  ',
                      footer='\n')
        doc = defaults_docstring(defaults, **kwargs)
        if func.__doc__ is None:
            func.__doc__ = ''
        func.__doc__ += doc
        return func

    return decorator


class Meta(type): #pragma: no cover
    """Meta class for appending docstring with defaults
    """
    def __new__(mcs, name, bases, attrs):
        attrs['_doc'] = attrs.get('__doc__', '')
        return super(Meta, mcs).__new__(mcs, name, bases, attrs)

    @property
    def __doc__(cls):
        kwargs = dict(header='Parameters\n----------\n',
                      indent='  ',
                      footer='\n')
        return cls._doc + cls.defaults_docstring(**kwargs)


class Property:
    """Base class for model properties.

    This class and its sub-classes implement variations on the concept
    of a 'mutable' value or 'l-value', i.e., an object that can be
    assigned a value.

    This class defines some interfaces that help read/write
    heirachical sets of properties between various formats
    (python dictionaries, yaml files, astropy tables, etc..)

    The pymodeler.model.Model class maps from property names to
    Property instances.

    """
    __metaclass__ = Meta

    __value__ = None

    defaults = [
        ('value', __value__, 'Property value'),
        ('help', "", 'Help description'),
        ('format', '%s', 'Format string for printing'),
        ('dtype', None, 'Data type'),
        ('default', None, 'Default value'),
        ('required', False, 'Is this propery required?'),
        ('unit', None, 'Units associated to value'),
    ]
        
    @defaults_decorator(defaults)
    def __init__(self, **kwargs):
        self._load(**kwargs)
        
    def __set_name__(self, owner, name):
        self.private_name = '_' + name
        
    def __str__(self):
        return self.format.format(self.__value__)

    def __set__(self, obj, value):
        cast_value = self.cast_type(value)
        return setattr(obj, self.private_name, cast_value)
        
    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __call__(self):
        """ __call__ will return the current value

        By default this invokes `self.__get__`
        so, any additional functionality that sub-classes implement,
        (such as caching the results of
        complicated operations needed to compute the value)
        will also be invoked
        """
        return self.__get__(self).__value__

    def __delete__(self):
        """Set the value to None

        This can be useful for sub-classes that use None
        to indicate an un-initialized value.

        Note that this invokes hooks for type-checking and
        bounds-checking that may be implemented by sub-classes, so it
        should will need to be re-implemented if those checks do note
        accept None as a valid value.
        """
        self.__value__ = None
    
    def _set_value(self, value):
        """Set the value

        This invokes hooks for type-checking and bounds-checking that
        may be implemented by sub-classes.
        """
        if is_none(value):
            value = None
        self.check_bounds(value)
        self.__value__ = self.cast_type(value)
      
    def _set_properties(self, **kwargs):
        """Set the value to kwargs['value']

        The invokes hooks for type-checking and bounds-checking that
        may be implemented by sub-classes.
        """
        if 'value' in kwargs:
            self._set_value(kwargs.pop('value'))
    
    def _load(self, **kwargs):
        """Load kwargs key,value pairs into __dict__
        """
        defaults = {d[0]:d[1] for d in self.defaults}
        # Require kwargs are in defaults
        for k in kwargs:
            if k not in defaults:
                msg = "Unrecognized attribute of %s: %s" % (self.__class__.__name__, k)
                raise AttributeError(msg)
        defaults.update(kwargs)

        # This doesn't overwrite the properties
        self.__dict__.update(defaults)

        # This should now be set
        default_value = self.cast_type(self.default)

        # This sets the underlying property values (i.e., __value__)
        self._set_value(default_value)

    @classmethod
    def defaults_docstring(cls, header=None, indent=None, footer=None):
        """Add the default values to the class docstring"""
        return defaults_docstring(cls.defaults, header=header,
                                  indent=indent, footer=footer)

    def innertype(self):
        """Return the type of the current value
        """
        return type(self.__value__)

    def todict(self):
        """Convert to a '~collections.OrderedDict' object.

        By default this only assigns {'value':self.value}
        """
        return odict(value=self.__str__())

    def dump(self):
        """Dump this object as a yaml string
        """
        return yaml.dump(self)

    def check_bounds(self, value):
        """Hook for bounds-checking, invoked during assignment.

        Sub-classes can raise an exception for out-of-bounds input values.
        """
        #pylint: disable=unused-argument, no-self-use
        return

    def cast_type(self, value):
        """Hook for type-checking, invoked during assignment.

        raises TypeError if neither value nor self.dtype are None and they
        do not match.

        will not raise an exception if either value or self.dtype is None
        """
        # if self.dtype is None, then any type is allowed
        if is_none(self.dtype):
            return value
        # value = None is always allowed
        if is_none(value):
            return None
        # if value is an instance of self.dtype, then return it
        if isinstance(value, self.dtype):
            return value
        # try and cast value to dtype
        try:
            return self.dtype(value)
        except (TypeError, ValueError):
            try:
                return self.dtype(*value)
            except (TypeError, ValueError):
                try:
                    return self.dtype(**value)
                except (TypeError, ValueError):
                    try:
                        return self.dtype(value['value'])
                    except:
                        pass
        msg = "Value of type %s, when %s was expected." % (type(value), self.__dict__['dtype'])
        raise TypeError(msg)   


class Derived(Property):
    """Property sub-class for derived model properties (i.e., properties
    that depend on other properties)

    This allows specifying the expected data type and formatting
    string for printing, and specifying a 'loader' function by name
    that is used to compute the value of the property.

    """

    defaults = deepcopy(Property.defaults) + [
        ('loader', None, 'Function to load datum')
    ]

    @defaults_decorator(defaults)
    def __init__(self, **kwargs):
        super(Derived, self).__init__(**kwargs)

    def __get__(self, obj, objtype=None):
        """Return the current value.

        This first checks if the value is cached (i.e., if
        `self.__value__` is not None)

        If it is not cached then it invokes the `load function to
        compute the value, and caches the computed value

        """
        if self.__value__ is None:
            try:
                loader = self.loader
            except KeyError as err: #pragma: no cover
                raise AttributeError("Loader is not defined") from err

            # Try to run the loader.
            # Don't catch expections here, let the Model class figure it out
            val = loader()

            # Try to set the value
            try:
                setattr(obj, self.private_name, val)
            except TypeError as err:
                msg = "Loader must return variable of type %s or None, got %s" % (obj.__dict__['dtype'], type(val))
                raise TypeError(msg) from err
        return getattr(obj, self.private_name)


class Model:
    
    def __init__(self, **kwargs):
        """ C'tor.  Build from a set of keyword arguments.
        """
        self._params = odict()
        self._find_properties()
        self._init_properties()
        self.set_attributes(**kwargs)
        # In case no properties were set, cache anyway
        self._cache()

    def _find_properties(self):
        the_classes = self.__class__.mro()
        for the_class in the_classes:
            for key, val in the_class.__dict__.items():
                if isinstance(val, Property):
                    self._params[key] = val
                else:
                    if key == 'optical_chain':
                        print("skipping %s" % key)
        
    def __str__(self, indent=0):
        """ Cast model as a formatted string
        """
        try:
            ret = '{0:>{2}}{1}'.format('', self.name, indent)
        except AttributeError:
            ret = "%s" % (type(self))
        if not self._params: #pragma: no cover
            pass
        else:
            ret += '\n{0:>{2}}{1}'.format('', 'Parameters:', indent + 2)
            width = len(max(self._params.keys(), key=len))
            for name, value in self._params.items():
                par = '{0!s:{width}} : {1!r}'.format(name, value, width=width)
                ret += '\n{0:>{2}}{1}'.format('', par, indent + 4)
        return ret

    #@property
    # def name(self):
    #    return self.__class__.__name__

    def getp(self, name):
        """
        Get the named `Property`.

        Parameters
        ----------
        name : str
            The property name.

        Returns
        -------
        param : `Property`
            The parameter object.

        """
        return self._params[name]

    def setp(self, name, **kwargs):
        """
        Set the value (and bounds) of the named parameter.

        Parameters
        ----------
        name : str
            The parameter name.

        Keywords
        --------
        clear_derived : bool
            Flag to clear derived objects in this model
        value:
            The value of the parameter, if None, it is not changed
        bounds: tuple or None
            The bounds on the parameter, if None, they are not set
        free : bool or None
            Flag to say if parameter is fixed or free in fitting, if None, it is not changed
        errors : tuple or None
            Uncertainties on the parameter, if None, they are not changed

        """
        kwcopy = kwargs.copy()
        clear_derived = kwcopy.pop('clear_derived', True)

        try:
            param = self._params[name]
            param.__set__(kwcopy)            
        except TypeError as msg:
            raise TypeError("Failed to set parameter %s" % name) from msg

        if clear_derived:
            self.clear_derived()
        self._cache(name)

    def set_attributes(self, **kwargs):
        """
        Set a group of attributes (parameters and members).  Calls
        `setp` directly, so kwargs can include more than just the
        parameter value (e.g., bounds, free, etc.).
        """
        self.clear_derived()
        kwargs = dict(kwargs)
        for name, value in kwargs.items():
            # Raise AttributeError if param not found
            try:
                self.getp(name)
            except KeyError:
                print ("Warning: %s does not have attribute %s" %
                       (type(self), name))
            # Set attributes
            try:
                self.setp(name, clear_derived=False, value=value)
            except (TypeError, KeyError):
                self.__setattr__(name, value)
            # pop this attribued off the list of missing properties
            self._missing.pop(name, None)
        # Check to make sure we got all the required properties
        if self._missing:
            raise ValueError(
                "One or more required properties are missing ",
                self._missing.keys())

    def _init_properties(self):
        """ Loop through the list of Properties,
        extract the derived and required properties and do the
        appropriate book-keeping
        """
        self._missing = {}
        for k, p in self._params.items():
            if p.required:
                self._missing[k] = p
            if isinstance(p, Derived):
                if p.loader is None:
                    # Default to using _<param_name>
                    p.loader = self.__getattribute__("_load_%s" % k)
                elif isinstance(p.loader, str):
                    p.loader = self.__getattribute__(p.loader)

    def get_params(self, pnames=None):
        """ Return a list of Parameter objects

        Parameters
        ----------

        pname : list or None
           If a list get the Parameter objects with those names

           If none, get all the Parameter objects

        Returns
        -------

        params : list
            list of Parameters

        """
        l = []
        if pnames is None:
            pnames = self._params.keys()
        for pname in pnames:
            p = self._params[pname]
            if isinstance(p, Parameter):
                l.append(p)
        return l

    def param_values(self, pnames=None):
        """ Return an array with the parameter values

        Parameters
        ----------

        pname : list or None
           If a list, get the values of the `Parameter` objects with those names

           If none, get all values of all the `Parameter` objects

        Returns
        -------

        values : `np.array`
            Parameter values

        """
        l = self.get_params(pnames)
        v = [p.value for p in l]
        return np.array(v)

    def param_errors(self, pnames=None):
        """ Return an array with the parameter errors

        Parameters
        ----------
        pname : list of string or none
           If a list of strings, get the Parameter objects with those names

           If none, get all the Parameter objects

        Returns
        -------
        ~numpy.array of parameter errors

        Note that this is a N x 2 array.
        """
        l = self.get_params(pnames)
        v = [p.errors for p in l]
        return np.array(v)

    def clear_derived(self):
        """ Reset the value of all Derived properties to None

        This is called by setp (and by extension __setattr__)
        """
        for p in self._params.values():
            if isinstance(p, Derived):
                del p

    def todict(self):
        """ Return self cast as an '~collections.OrderedDict' object
        """
        ret = odict(name=self.__class__.__name__)
        for key in self._params.keys():
            ret.update({key:getattr(self,key)})
        return ret

    def dump(self):
        """ Dump this object as a yaml string
        """
        return yaml.dump(self.todict())

    def _cache(self, name=None):
        """
        Method called in _setp to cache any computationally
        intensive properties after updating the parameters.

        Parameters
        ----------
        name : string
           The parameter name.

        Returns
        -------
        None
        """
        #pylint: disable=unused-argument, no-self-use
        return
