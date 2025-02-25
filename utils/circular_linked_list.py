# Node class to represent each element in the linked list
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None 
        self.prev = None

# Circular LinkedList class to manage the nodes
class CLinkedList:
    def __init__(self, init_list):
        self.head = None  # Head of the linked list
        for val in init_list:
            self.append(val)
        self.print_list()

    # Method to add a node at the end of the circular linked list
    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            new_node.next = new_node
            new_node.prev = new_node
            return
        # Traverse to the last node
        last_node = self.head.prev  # Last node is head's previous
        last_node.next = new_node
        new_node.prev = last_node
        new_node.next = self.head
        self.head.prev = new_node

    def print_list(self):
        print("Circular Linked List: ")
        if self.head is None:
            print("The list is empty")
            return
        current_node = self.head
        while True:
            print(current_node.data, end=" -> ")
            current_node = current_node.next
            if current_node == self.head:
                break
        print(" initialized \n")

# Example usage
if __name__ == "__main__":
    llist = CLinkedList([1,2,3,4,5,6])
    # llist.print_list()  # Output: 0 -> 1 -> 1.5 -> 2 -> 3 -> None