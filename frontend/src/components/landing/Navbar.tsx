// PlatosCave/frontend/src/components/landing/Navbar.tsx
import React from 'react';
import { ArrowRight, Menu } from 'lucide-react';
import platosCaveLogo from '../../images/platos-cave-logo.png';

interface NavbarProps {
  onGetStarted?: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ onGetStarted }) => {
  const navLinks = [
    { label: 'How it works', href: '#how-it-works' },
    { label: 'Models', href: '#models' },
    { label: 'Pricing', href: '#pricing' },
    { label: 'About', href: '#about' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        {/* Logo */}
        <a 
          href="/" 
          className="flex items-center gap-2 transition-opacity duration-200 hover:opacity-80"
        >
          <img 
            src={platosCaveLogo} 
            alt="Plato's Cave" 
            className="h-9 w-auto"
          />
        </a>

        {/* Navigation Links - Hidden on mobile */}
        <div className="hidden items-center gap-8 md:flex">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="text-sm font-medium text-text-secondary transition-colors duration-200 hover:text-text-primary"
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* CTA Button */}
        <button
          onClick={onGetStarted}
          className="group hidden md:inline-flex items-center gap-2 rounded-full bg-text-primary px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 hover:shadow-lg active:scale-[0.98]"
        >
          Get Started
          <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
        </button>

        {/* Mobile Menu Button */}
        <button className="rounded-lg p-2 text-text-secondary transition-colors hover:bg-gray-100 md:hidden">
          <Menu className="h-6 w-6" />
        </button>
      </nav>
    </header>
  );
};

export default Navbar;
