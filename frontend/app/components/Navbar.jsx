import { useState } from 'react';
import { Link } from 'react-router'; // keep this if you're using react-router with RouterProvider

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  return (
    <nav className="navbar flex items-center justify-between px-4 py-3 relative">
      {/* Logo */}
      <Link to='/' className='text-2xl font-bold text-gradient'>
        LENNI
      </Link>

      {/* Hamburger Button (Mobile Only) */}
      <button
        className="sm:hidden text-black focus:outline-none"
        onClick={toggleMenu}
        aria-label="Toggle menu"
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          {isOpen ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M6 18L18 6M6 6l12 12"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M4 6h16M4 12h16M4 18h16"
            />
          )}
        </svg>
      </button>

      {/* Nav Links */}
      <div
        className={`${
          isOpen ? 'block' : 'hidden'
        } absolute top-full left-0 w-full bg-white shadow-md sm:shadow-none sm:bg-transparent sm:static sm:flex sm:items-center sm:space-x-6 sm:w-auto`}
      >
        <Link to="#about" className="block px-4 py-2 font-bold text-black text-[1.1rem] sm:inline-block">
          About
        </Link>
        <Link to="#implementation" className="block px-4 py-2 font-bold text-black text-[1.1rem] sm:inline-block">
          Implementation
        </Link>
        <Link to="#support" className="block px-4 py-2 font-bold text-black text-[1.1rem] sm:inline-block">
          Support
        </Link>
        <Link to="/start" className="block px-4 py-2 font-bold text-black text-[1.1rem] sm:inline-block">
          Talk to Lenni
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
