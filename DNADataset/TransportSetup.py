import numpy as np
import os
from collections import defaultdict, OrderedDict

class DNAPDBParser:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.atoms = []
        self.strands = []  # List of lists of residues
        self.method = 'B3LYP'
        
        # Orbital mappings for B3LYP method
        self.orbital_map = {
            'H': 5, 'C': 15, 'N': 15, 'O': 15, 'P': 19, 'S': 19,
            'Na': 19, 'Fe': 24, 'Ag': 24, 'Hg': 20
        }
        
    def parse_pdb(self):
        """Parse PDB file respecting TER records to separate strands"""
        self.atoms = []
        self.strands = []
        
        current_strand = defaultdict(list)
        
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_info = self._parse_atom_line(line)
                    if atom_info:
                        self.atoms.append(atom_info)
                        
                        # Group by residue number within current strand
                        res_id = atom_info['res_num']
                        current_strand[res_id].append(atom_info)
                        
                elif line.startswith('TER'):
                    # End of current strand
                    if current_strand:
                        # Convert to ordered dict and add to strands
                        sorted_strand = OrderedDict(sorted(current_strand.items()))
                        self.strands.append(sorted_strand)
                        current_strand = defaultdict(list)
        
        # Add final strand if exists
        if current_strand:
            sorted_strand = OrderedDict(sorted(current_strand.items()))
            self.strands.append(sorted_strand)
    
    def _parse_atom_line(self, line):
        """Parse a single ATOM line from PDB"""
        try:
            # Get element - try column 76-78 first, fallback to atom name
            element = line[76:78].strip() if len(line) > 76 and line[76:78].strip() else ''
            if not element:
                # Extract element from atom name (first 1-2 chars, removing digits)
                atom_name = line[12:16].strip()
                element = ''.join([c for c in atom_name if c.isalpha()])[:2]
                # Handle common cases
                if element.startswith('H'):
                    element = 'H'
                elif len(element) > 1 and element[1].islower():
                    element = element[0] + element[1]
                else:
                    element = element[0]
            
            atom_info = {
                'atom_num': int(line[6:11].strip()),
                'atom_name': line[12:16].strip(),
                'res_name': line[17:20].strip(),
                'res_num': int(line[22:26].strip()),
                'x': float(line[30:38].strip()),
                'y': float(line[38:46].strip()),
                'z': float(line[46:54].strip()),
                'element': element
            }
            return atom_info
        except (ValueError, IndexError) as e:
            return None
    
    def get_contact_atoms(self, strand_selection='cross_strand'):
        """
        Get atoms for left and right contacts (always full base)
        
        strand_selection: 
            - 'cross_strand': first base (5' end) of each strand (default, for measuring across strands)
            - 'same_strand': both ends (5' and 3') of first strand
        """
        if not self.strands:
            raise ValueError("No strands found. Run parse_pdb() first.")
        
        contacts = {'left': [], 'right': []}
        
        if strand_selection == 'cross_strand':
            # Left contact: first base (5' end) of first strand
            # Right contact: first base (5' end) of second strand
            left_strand = self.strands[0]
            right_strand = self.strands[-1] if len(self.strands) > 1 else self.strands[0]
            
            # All atoms in first residue of each strand
            first_res_left = list(left_strand.keys())[0]
            contacts['left'] = [atom['atom_num'] for atom in left_strand[first_res_left]]
            
            first_res_right = list(right_strand.keys())[0]
            contacts['right'] = [atom['atom_num'] for atom in right_strand[first_res_right]]
        
        elif strand_selection == 'same_strand':
            # Both contacts on first strand (5' and 3' ends)
            strand = self.strands[0]
            
            first_res = list(strand.keys())[0]
            last_res = list(strand.keys())[-1]
            contacts['left'] = [atom['atom_num'] for atom in strand[first_res]]
            contacts['right'] = [atom['atom_num'] for atom in strand[last_res]]
        
        return contacts
    
    def generate_parameters_file(self, output_file='Parameters.txt', 
                                 strand_selection='cross_strand',
                                 energy_range=(-7, 0.01, 0.01), 
                                 gamma_l=0.1, 
                                 gamma_r=0.1,
                                 broadening=0.0):
        """Generate Parameters.txt file for transmission calculation"""
        
        # Parse PDB file
        self.parse_pdb()
        
        # Get structure name from PDB filename
        structure_name = os.path.splitext(os.path.basename(self.pdb_file))[0]
        
        # Calculate orbitals for each atom
        orbitals = []
        for atom in self.atoms:
            element = atom['element']
            if element in self.orbital_map:
                orbitals.append(self.orbital_map[element])
            else:
                raise ValueError(f"Unrecognized element: {element} (atom {atom['atom_num']})")
        
        # Generate energy range
        start, stop, step = energy_range
        energy_values = np.arange(start, stop, step)
        
        # Get contact atoms
        contacts = self.get_contact_atoms(strand_selection)
        
        # Write Parameters.txt file
        with open(output_file, 'w') as f:
            f.write(f"{structure_name}\n\n")
            
            f.write("Orbitals set\n")
            for orbital in orbitals:
                f.write(f"{orbital:.6f}\n")
            f.write("\n")
            
            f.write("Energy Range\n")
            for energy in energy_values:
                f.write(f"{energy:.6f}\n")
            f.write("\n")
            
            f.write("Inject Site (atoms number)\n")
            for atom_num in contacts['left']:
                f.write(f"{atom_num:.6f}\n")
            f.write("\n")
            
            f.write("Extract Site (atoms number)\n")
            for atom_num in contacts['right']:
                f.write(f"{atom_num:.6f}\n")
            f.write("\n")
            
            f.write("GammaL\n")
            f.write(f"{gamma_l:.6f}\n\n")
            
            f.write("GammaR\n")
            f.write(f"{gamma_r:.6f}\n\n")
            
            f.write("Probes Site (atoms number)\n\n")
            
            f.write("Broadening (for DOS)\n")
            f.write(f"{broadening:.6f}\n\n")
            
            f.write("Probe (for Decoh)\n")
            f.write("0.010000\n")  # Add a default probe value
        
        return contacts, len(self.atoms)
    
    def print_structure_info(self):
        """Print information about the DNA structure"""
        self.parse_pdb()
        
        print(f"Structure: {os.path.basename(self.pdb_file)}")
        print(f"Total atoms: {len(self.atoms)}")
        print(f"Number of strands: {len(self.strands)}")
        
        for idx, strand in enumerate(self.strands):
            print(f"\nStrand {idx + 1}:")
            print(f"  Residues: {len(strand)}")
            print(f"  Residue numbers: {list(strand.keys())}")
            
            # Get sequence
            sequence = [strand[res][0]['res_name'] for res in strand.keys()]
            print(f"  Sequence: {' '.join(sequence)}")
            
            # First and last atom numbers
            first_res = list(strand.keys())[0]
            last_res = list(strand.keys())[-1]
            first_atom = strand[first_res][0]['atom_num']
            last_atom = strand[last_res][-1]['atom_num']
            print(f"  Atom range: {first_atom} - {last_atom}")
            
            # Show first base info (5' end)
            print(f"  5' end (first base, residue {first_res}): atoms {strand[first_res][0]['atom_num']}-{strand[first_res][-1]['atom_num']} ({len(strand[first_res])} atoms)")
            print(f"  3' end (last base, residue {last_res}): atoms {strand[last_res][0]['atom_num']}-{strand[last_res][-1]['atom_num']} ({len(strand[last_res])} atoms)")


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(
        description='Generate Parameters.txt file for DNA transmission calculations from PDB file'
    )
    arg_parser.add_argument(
        'filename', 
        type=str,
        help='Input PDB filename'
    )
    arg_parser.add_argument(
        '--mode',
        type=str,
        choices=['same', 'cross'],
        default='same',
        help='Contact mode: "same" for same-strand (5\' to 3\'), "cross" for cross-strand (5\' to 5\') (default: same)'
    )
    arg_parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Gamma coupling intensity for both contacts (default: 0.1)'
    )
    
    args = arg_parser.parse_args()
    
    # Map mode to strand_selection
    mode_map = {
        'same': 'same_strand',
        'cross': 'cross_strand'
    }
    strand_selection = mode_map[args.mode]
    
    # Get base filename without extension for output
    base_filename = os.path.splitext(os.path.basename(args.filename))[0]
    output_filename = f"Parameters_{base_filename}.txt"
    
    # Initialize parser
    parser = DNAPDBParser(args.filename)
    
    # Print structure information
    print("=== DNA Structure Information ===")
    parser.print_structure_info()
    
    print(f"\n=== Generating Parameters File ===\n")
    
    try:
        contacts, total_atoms = parser.generate_parameters_file(
            output_file=output_filename,
            strand_selection=strand_selection,
            energy_range=(-7, 0.01, 0.01),
            gamma_l=args.gamma,
            gamma_r=args.gamma
        )
        
        mode_desc = "Same strand: 5' end → 3' end" if args.mode == 'same' else "Cross-strand: 5' end of strand 1 → 5' end of strand 2"
        
        print(f"Output: {output_filename}")
        print(f"  Mode: {mode_desc}")
        print(f"  Gamma: {args.gamma}")
        print(f"  Left contact: {len(contacts['left'])} atoms: {contacts['left']}")
        print(f"  Right contact: {len(contacts['right'])} atoms: {contacts['right']}")
        print(f"  Total atoms in system: {total_atoms}")
        print(f"\nParameters file generated successfully!")
        
    except Exception as e:
        print(f"Error generating parameters file: {e}")
